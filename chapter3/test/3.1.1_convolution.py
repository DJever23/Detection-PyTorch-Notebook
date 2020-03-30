import torch
from torch import nn
# 使用torch.nn中的Conv2d()搭建卷积层
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1,groups=1, bias=True)
'''
对于torch.nn.Conv2d()来说,传入的参数含义如下:
·in_channels:输入特征图的通道数,如果是RGB图像,则通道数为3。卷积中的特征图通道数一般是2的整数次幂。
·out_channels:输出特征图的通道数。
·kernel_size:卷积核的尺寸,常见的有1、3、5、7。
·stride:步长,即卷积核在特征图上滑动的步长,一般为1。如果大于1,则输出特征图的尺寸会小于输入特征图的尺寸。
·padding:填充,常见的有零填充、边缘填充等,PyTorch默认为零填充。
·dilation:空洞卷积,当大于1时可以增大感受野的同时保持特征图的尺寸(后面会细讲),默认为1。
·groups:可实现组卷积,即在卷积操作时不是逐点卷积,而是将输入通道分为多个组,稀疏连接达到降低计算量的目的(后续会细讲),默认为1。
·bias:是否需要偏置,默认为True。
'''
# 查看卷积核的基本信息，本质上是一个Module
print('conv', conv)  # Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# 通过.weight与.bias查看卷积核的权重与偏置
print('conv.weight.shape', conv.weight.shape)  # torch.Size([1, 1, 3, 3])
print('conv.bias.shape', conv.bias.shape)  # torch.Size([1])
# 输入特征图，需要注意特征必须是四维，第一维作为batch数，即使是1也要保留
input = torch.ones(1, 1, 5, 5)
output = conv(input)
# 当前配置的卷积核可以使输入和输出的大小一致
print('input.shape', input.shape)  # torch.Size([1, 1, 5, 5])
print('output.shape', output.shape)  # torch.Size([1, 1, 5, 5])
