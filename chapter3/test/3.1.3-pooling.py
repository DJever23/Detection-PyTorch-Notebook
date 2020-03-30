import torch
from torch import nn
# 池化主要需要两个参数，第一个参数代表池化区域大小，第二个参数表示步长
max_pooling = nn.MaxPool2d(2, stride=2)
aver_pooling = nn.AvgPool2d(2, stride=2)
input = torch.randn(1, 1, 4, 4)
print('input', input)
# 调用最大池化与平均池化，可以看到size从[1,1,4,4]变成了[1,1,2,2]
print('max', max_pooling(input))
print('avg', aver_pooling(input))