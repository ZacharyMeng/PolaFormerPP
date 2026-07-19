# bash tools/dist_train.sh configs/PolaPP/polapp_b1.py 8 \
#     --options model.pretrained=path/to/polapp_b1.pth

# bash tools/dist_train.sh configs/PolaPP/polapp_b2.py 8 \
#     --options model.pretrained=path/to/polapp_b2.pth

bash tools/dist_train.sh configs/PolaPP/polapp_b3.py 8 \
    --options model.pretrained=path/to/polapp_b3.pth
