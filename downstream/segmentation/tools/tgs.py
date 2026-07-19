

import argparse
import os
import time
from contextlib import nullcontext

import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel, scatter
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')

    # 必要参数
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')

    # 速度相关
    parser.add_argument(
        '--aug-test', action='store_true',
        help='Use Flip and Multi scale aug', default=False)

    # 原有评测/输出
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only', action='store_true',
        help=('Format the output results without perform evaluation. '
              'Useful for specific format to submit to the test server'))
    parser.add_argument(
        '--eval', type=str, nargs='+',
        help=('evaluation metrics, depends on the dataset, e.g., "mIoU" '
              'for generic datasets, and "cityscapes" for Cityscapes'))
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show-dir', help='directory to save painted images')
    parser.add_argument(
        '--gpu-collect', action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
             'workers, available when gpu_collect is not specified')
    parser.add_argument('--options', nargs='+', action=DictAction,
                        help='custom options')
    parser.add_argument('--eval-options', nargs='+', action=DictAction,
                        help='custom options for evaluation')

    # 启动方式
    parser.add_argument(
        '--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    # 叠图透明度（保留原参数）
    parser.add_argument(
        '--opacity', type=float, default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')

    # ===== 新增：吞吐测试与加速相关选项 =====
    parser.add_argument('--throughput', action='store_true',
                        help='Measure throughput only (no eval/format/show).')
    parser.add_argument('--warmup-iters', type=int, default=10,
                        help='Warmup iterations before timing.')
    parser.add_argument('--max-iters', type=int, default=-1,
                        help='Max timed iterations; -1 means run full dataloader once.')
    parser.add_argument('--channels-last', action='store_true',
                        help='Use channels_last (NHWC) memory format for model & inputs.')
    parser.add_argument('--fp16-infer', action='store_true',
                        help='Force FP16 autocast inference regardless of cfg.fp16.')

    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


# --------- 工具函数 ---------
def to_channels_last(sample):
    """将 4D Tensor 递归转为 channels_last；支持 dict/list/tuple。"""
    if torch.is_tensor(sample):
        return sample.to(memory_format=torch.channels_last) if sample.dim() == 4 else sample
    if isinstance(sample, (list, tuple)):
        t = [to_channels_last(v) for v in sample]
        return type(sample)(t) if isinstance(sample, tuple) else t
    if isinstance(sample, dict):
        return {k: to_channels_last(v) for k, v in sample.items()}
    return sample


def get_batch_size(data):
    """尽量稳健地拿到 batch size。优先从 data['img'] 取，其次任意张量。"""
    if isinstance(data, dict):
        if 'img' in data:
            v = data['img']
            if torch.is_tensor(v):
                return v.size(0)
            if isinstance(v, (list, tuple)) and v and torch.is_tensor(v[0]):
                return v[0].size(0)
        for v in data.values():
            if torch.is_tensor(v) and v.dim() > 0:
                return v.size(0)
            if isinstance(v, (list, tuple)) and v and torch.is_tensor(v[0]):
                return v[0].size(0)
    return 1


# --------- 吞吐统计（单卡）---------
def throughput_single_plain(model, data_loader, fp16=False, use_channels_last=False,
                            warmup_iters=10, max_iters=-1):
    """单卡吞吐：模型为普通 nn.Module（非 DP/DDP）。使用 mmcv.scatter 处理 DataContainer。"""
    model.eval()
    device = torch.cuda.current_device()
    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if fp16 else nullcontext()

    # warmup
    it = 0
    with torch.inference_mode(), amp_ctx:
        for data in data_loader:
            # 关键：展开 DataContainer 并搬到当前 GPU
            data = scatter(data, [device])[0]
            if use_channels_last and 'img' in data:
                data['img'] = to_channels_last(data['img'])
            _ = model(return_loss=False, **data)
            it += 1
            if it >= warmup_iters:
                break
    torch.cuda.synchronize()

    # # timed
    # total_imgs = 0
    # it = 0
    # start = time.perf_counter()
    
    # with torch.inference_mode(), amp_ctx:
    #     for data in data_loader:
    #         data = scatter(data, [device])[0]
    #         if use_channels_last and 'img' in data:
    #             data['img'] = to_channels_last(data['img'])
    #         _ = model(return_loss=False, **data)
    #         total_imgs += get_batch_size(data)
    #         it += 1
    #         if max_iters > 0 and it >= max_iters:
    #             break
    # torch.cuda.synchronize()
    # elapsed = time.perf_counter() - start
    # ips = total_imgs / elapsed if elapsed > 0 else 0.0
    # print(f'[Throughput][Single GPU] images: {total_imgs}, time: {elapsed:.4f}s, imgs/s: {ips:.2f}')
    # return ips

    # timed —— 只量纯前向
    total_imgs = 0
    it = 0
    elapsed_ms = 0.0  # 累计前向耗时（毫秒）

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    with torch.inference_mode(), amp_ctx:
        for data in data_loader:
            print(it)
            # 前处理与搬运不计时
            data = scatter(data, [device])[0]
            if use_channels_last and 'img' in data:
                data['img'] = to_channels_last(data['img'])

            # --- 开始只测这一步 ---
            torch.cuda.synchronize()        # 防止前一轮遗留的异步干扰
            start_evt.record()
            _ = model(return_loss=False, **data)
            end_evt.record()
            torch.cuda.synchronize()        # 等这次前向结束
            iter_ms = start_evt.elapsed_time(end_evt)  # 毫秒
            # --- 只测这一步结束 ---

            elapsed_ms += iter_ms
            total_imgs += get_batch_size(data)
            it += 1
            if max_iters > 0 and it >= max_iters:
                break

    elapsed_s = elapsed_ms / 1000.0
    ips = total_imgs / elapsed_s if elapsed_s > 0 else 0.0
    print(f'[Throughput(FWD-only)][Single GPU] images: {total_imgs}, '
        f'model_time: {elapsed_s:.4f}s, imgs/s: {ips:.2f}, '
        f'avg_batch_ms: {elapsed_ms/it:.3f}')
    return ips


# --------- 吞吐统计（多卡 DDP）---------
def throughput_distributed_ddp(model, data_loader, fp16=False, use_channels_last=False,
                               warmup_iters=10, max_iters=-1):
    """多卡吞吐：模型为 MMDistributedDataParallel。使用 scatter 处理 DataContainer。"""
    model.eval()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.cuda.current_device()
    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if fp16 else nullcontext()

    # warmup
    it = 0
    with torch.inference_mode(), amp_ctx:
        for data in data_loader:
            data = scatter(data, [device])[0]
            if use_channels_last and 'img' in data:
                data['img'] = to_channels_last(data['img'])
            _ = model(return_loss=False, **data)
            it += 1
            if it >= warmup_iters:
                break
    torch.cuda.synchronize()
    dist.barrier()

    # timed
    local_imgs = 0
    it = 0
    start = time.perf_counter()
    with torch.inference_mode(), amp_ctx:
        for data in data_loader:
            data = scatter(data, [device])[0]
            if use_channels_last and 'img' in data:
                data['img'] = to_channels_last(data['img'])
            _ = model(return_loss=False, **data)
            local_imgs += get_batch_size(data)
            it += 1
            if max_iters > 0 and it >= max_iters:
                break
    torch.cuda.synchronize()
    local_time = time.perf_counter() - start

    # 汇总：图像总数求和，时间取最大
    total_imgs_tensor = torch.tensor([float(local_imgs)], dtype=torch.float64, device='cuda')
    total_time_tensor = torch.tensor([float(local_time)], dtype=torch.float64, device='cuda')
    dist.all_reduce(total_imgs_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_time_tensor, op=dist.ReduceOp.MAX)
    total_imgs = int(total_imgs_tensor.item())
    total_time = float(total_time_tensor.item())
    ips = total_imgs / total_time if total_time > 0 else 0.0
    if rank == 0:
        print(f'[Throughput][Distributed x{world_size}] images: {total_imgs}, '
              f'time(max): {total_time:.4f}s, imgs/s: {ips:.2f}')
    return ips


def main():
    args = parse_args()

    # 吞吐模式不强制要求 out/eval/show；普通模式保持原断言
    if not args.throughput:
        assert args.out or args.eval or args.format_only or args.show or args.show_dir, \
            ('Please specify at least one operation (save/eval/format/show the results / save the results) '
             'with the argument "--out", "--eval", "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    # cuDNN benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # aug-test
    if args.aug_test:
        # hard code index (沿用原写法)
        cfg.data.test.pipeline[1].img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        cfg.data.test.pipeline[1].flip = True

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # 分布式初始化
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # 数据集与 DataLoader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False
    )
    # 尽量启用 pin_memory / persistent_workers（若可用）
    if hasattr(data_loader, 'pin_memory'):
        try:
            data_loader.pin_memory = True
        except Exception:
            pass
    if hasattr(data_loader, 'persistent_workers'):
        try:
            data_loader.persistent_workers = True
        except Exception:
            pass

    # 构建与加载模型
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'meta' in checkpoint:
        meta = checkpoint['meta']
        if 'CLASSES' in meta:
            model.CLASSES = meta['CLASSES']
        if 'PALETTE' in meta:
            model.PALETTE = meta['PALETTE']

    # 可选：将模型设为 channels_last
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # efficient_test（省显存，不等价于更快）
    efficient_test = False
    if args.eval_options is not None:
        efficient_test = args.eval_options.get('efficient_test', False)

    # ===== 吞吐模式 =====
    if args.throughput:
        if not distributed:
            # 单卡：直接把模型搬到 GPU，绕过 DataParallel 的 scatter 开销
            model = model.cuda()
            throughput_single_plain(
                model=model,
                data_loader=data_loader,
                fp16=args.fp16_infer,
                use_channels_last=args.channels_last,
                warmup_iters=args.warmup_iters,
                max_iters=args.max_iters
            )
        else:
            # 多卡：用 DDP（注意此处模型需包一层 DDP）
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False
            )
            throughput_distributed_ddp(
                model=model,
                data_loader=data_loader,
                fp16=args.fp16_infer,
                use_channels_last=args.channels_last,
                warmup_iters=args.warmup_iters,
                max_iters=args.max_iters
            )
        return  # 吞吐模式到此结束

    # ===== 非吞吐模式：沿用原逻辑评测/导出/显示 =====
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  efficient_test, args.opacity)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect, efficient_test)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            dataset.evaluate(outputs, args.eval, **kwargs)


if __name__ == '__main__':
    main()
