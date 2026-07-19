# RetinaNet 1x
# bash tools/dist_train.sh configs/PolaPP/retinanet_b2_1x.py 8 \
#     --cfg-options model.pretrained=path/to/polapp_b2.pth
# bash tools/dist_train.sh configs/PolaPP/retinanet_b3_1x.py 8 \
#     --cfg-options model.pretrained=path/to/polapp_b3.pth

# Mask R-CNN 1x
# bash tools/dist_train.sh configs/PolaPP/maskrcnn_b2_1x.py 8 \
#     --cfg-options model.pretrained=path/to/polapp_b2.pth
# bash tools/dist_train.sh configs/PolaPP/maskrcnn_b3_1x.py 8 \
#     --cfg-options model.pretrained=path/to/polapp_b3.pth
# bash tools/dist_train.sh configs/PolaPP/maskrcnn_b4_1x.py 8 \
#     --cfg-options model.pretrained=path/to/polapp_b4.pth
# bash tools/dist_train.sh configs/PolaPP/maskrcnn_b5_1x.py 8 \
#     --cfg-options model.pretrained=path/to/polapp_b5.pth

# Mask R-CNN 3x
# bash tools/dist_train.sh configs/PolaPP/maskrcnn_b2_3x.py 8 \
#     --cfg-options model.pretrained=path/to/polapp_b2.pth
# bash tools/dist_train.sh configs/PolaPP/maskrcnn_b3_3x.py 8 \
#     --cfg-options model.pretrained=path/to/polapp_b3.pth
bash tools/dist_train.sh configs/PolaPP/maskrcnn_b4_3x.py 8 \
    --cfg-options model.pretrained=path/to/polapp_b4.pth
