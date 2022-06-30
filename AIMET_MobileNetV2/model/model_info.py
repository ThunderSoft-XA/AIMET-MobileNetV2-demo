import torchvision
from torch import nn

import MobileNetV2

net = MobileNetV2.mobilenet_v2(pretrained=True)
print(net)

params=net.state_dict()
len(params)
for i, j in enumerate(params):
    print (i, j)
# modelDict = {name: param.data for name, param in net.named_parameters()}
# print(modelDict)

