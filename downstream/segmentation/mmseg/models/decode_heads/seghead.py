import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead

from efficientvit.models.nn import (
    MBConv,
    FusedMBConv,
    ResidualBlock,
    IdentityLayer,
)


@HEADS.register_module()
class EfficientViTHead(BaseDecodeHead):
    """SegHead 改造版: 兼容 MMSeg BaseDecodeHead
    Args:
        head_stride (int): 对齐尺度
        head_width (int): feature 融合后的宽度
        head_depth (int): 中间 block 堆叠深度
        expand_ratio (float): MBConv/FMBConv expand ratio
        middle_op (str): "mbconv" or "fmbconv"
        final_expand (float | None): 是否再做一次通道扩展
        strides (list[int]): 每个输入 feature 的 stride
    """

    def __init__(self,
                 head_stride=8,
                 head_width=64,
                 head_depth=3,
                 expand_ratio=4,
                 middle_op="mbconv",
                 final_expand=None,
                 strides=[8, 16, 32],
                 act_cfg=dict(type='ReLU'),
                 **kwargs):
        super(EfficientViTHead, self).__init__(
            input_transform='multiple_select',
            **kwargs)

        self.head_stride = head_stride
        self.head_width = head_width
        self.head_depth = head_depth
        self.expand_ratio = expand_ratio
        self.middle_op = middle_op
        self.final_expand = final_expand
        self.strides = strides

        # 将不同 stage 的 feature 投影到 head_width
        self.proj_convs = nn.ModuleList()
        for in_ch, stride in zip(self.in_channels, self.strides):
            factor = stride // head_stride
            ops = [ConvModule(in_ch, head_width, 1,
                              norm_cfg=self.norm_cfg,
                              act_cfg=None)]
            if factor > 1:
                ops.append(nn.Upsample(
                    scale_factor=factor, mode="bilinear",
                    align_corners=self.align_corners))
            self.proj_convs.append(nn.Sequential(*ops))

        # 中间 block 堆叠
        middle = []
        for _ in range(head_depth):
            if middle_op == "mbconv":
                block = MBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm="bn2d",   # ⚠️ 这里用 EfficientViT 自己的 norm 定义
                    act_func=("relu", "relu", None),
                )
            elif middle_op == "fmbconv":
                block = FusedMBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm="bn2d",
                    act_func=("relu", None),
                )
            else:
                raise NotImplementedError
            middle.append(ResidualBlock(block, IdentityLayer()))
        self.middle = nn.Sequential(*middle)

        # 输出分类头
        out_channels = head_width if final_expand is None else int(head_width * final_expand)
        self.out_conv = nn.Sequential(
            nn.Conv2d(head_width, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) if act_cfg is not None else nn.Identity(),
            nn.Conv2d(out_channels, self.num_classes, kernel_size=1)
        )

    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)  # 根据 in_index 取出 backbone 对应层

        # 将所有输入 resize 到相同尺度并投影
        feats = []
        for x, proj in zip(inputs, self.proj_convs):
            feats.append(proj(x))
        x = sum(feats)  # add 融合

        # 中间 block
        x = self.middle(x)

        # 输出 logits
        output = self.out_conv(x)
        output = resize(
            input=output,
            size=inputs[0].shape[2:],  # 对齐到最浅层输入特征
            mode='bilinear',
            align_corners=self.align_corners
        )
        return output


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmcv.cnn import ConvModule

# from mmseg.ops import resize
# from ..builder import HEADS
# from .decode_head import BaseDecodeHead

# from efficientvit.models.nn import (
#     MBConv,
#     FusedMBConv,
#     ResidualBlock,
#     IdentityLayer,
# )


# @HEADS.register_module()
# class EfficientViTHead(BaseDecodeHead):
#     """SegHead 改造版: 兼容 MMSeg BaseDecodeHead
#     Args:
#         head_stride (int): 对齐尺度
#         head_width (int): feature 融合后的宽度
#         head_depth (int): 中间 block 堆叠深度
#         expand_ratio (float): MBConv/FMBConv expand ratio
#         middle_op (str): "mbconv" or "fmbconv"
#         final_expand (float | None): 是否再做一次通道扩展
#     """

#     def __init__(self,
#                  head_stride=8,
#                  head_width=64,
#                  head_depth=3,
#                  expand_ratio=4,
#                  middle_op="mbconv",
#                  final_expand=None,
#                  act_cfg=dict(type='ReLU'),
#                  **kwargs):
#         super(EfficientViTHead, self).__init__(
#             input_transform='multiple_select',
#             **kwargs)

#         self.head_stride = head_stride
#         self.head_width = head_width
#         self.head_depth = head_depth
#         self.expand_ratio = expand_ratio
#         self.middle_op = middle_op
#         self.final_expand = final_expand

#         # 将不同 stage 的 feature 投影到 head_width
#         self.proj_convs = nn.ModuleList()
#         for in_ch, stride in zip(self.in_channels, self.strides):
#             factor = stride // head_stride
#             ops = [ConvModule(in_ch, head_width, 1, norm_cfg=self.norm_cfg, act_cfg=None)]
#             if factor > 1:
#                 ops.append(nn.Upsample(scale_factor=factor, mode="bilinear", align_corners=self.align_corners))
#             self.proj_convs.append(nn.Sequential(*ops))

#         # 中间 block 堆叠
#         middle = []
#         for _ in range(head_depth):
#             if middle_op == "mbconv":
#                 block = MBConv(
#                     head_width,
#                     head_width,
#                     expand_ratio=expand_ratio,
#                     norm="bn2d",
#                     act_func=("relu", "relu", None),
#                 )
#             elif middle_op == "fmbconv":
#                 block = FusedMBConv(
#                     head_width,
#                     head_width,
#                     expand_ratio=expand_ratio,
#                     norm="bn2d",
#                     act_func=("relu", None),
#                 )
#             else:
#                 raise NotImplementedError
#             middle.append(ResidualBlock(block, IdentityLayer()))
#         self.middle = nn.Sequential(*middle)

#         # 输出分类头
#         out_channels = head_width if final_expand is None else int(head_width * final_expand)
#         self.out_conv = nn.Sequential(
#             nn.Conv2d(head_width, out_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True) if act_cfg is not None else nn.Identity(),
#             nn.Conv2d(out_channels, self.num_classes, kernel_size=1)
#         )

#     def forward(self, inputs):
#         """Forward function."""
#         inputs = self._transform_inputs(inputs)  # list of feature maps

#         # 将所有输入 resize 到相同尺度并投影
#         feats = []
#         for x, proj in zip(inputs, self.proj_convs):
#             feats.append(proj(x))
#         x = sum(feats)  # add 融合

#         # 中间 block
#         x = self.middle(x)

#         # 输出 logits
#         output = self.out_conv(x)
#         output = resize(
#             input=output,
#             size=inputs[0].shape[2:],  # 恢复到 backbone 最浅层的分辨率
#             mode='bilinear',
#             align_corners=self.align_corners
#         )
#         return output