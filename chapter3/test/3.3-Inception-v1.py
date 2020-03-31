import torch
from torch import nn
'''
Inception v1(又名GoogLeNet)网络较好地解决了由于增加神经网络的深度容易带来的梯度消失的现象。
Inception v1网络是一个精心设计的22层卷积网络,并提出了具有良好局部特征结构的Inception模块,即对特征并行地执行多个
大小不同的卷积运算与池化,最后再拼接到一起。由于1×1、3×3和5×5的卷积运算对应不同的特征图区域,因此这样做的好处是可以得到更好的图像表征信息。

为进一步降低网络参数量,Inception又增加了多个1×1的卷积模块。这种1×1的模块可以先将特征图降维,再送给3×3和5×5大小的卷积核,
由于通道数的降低,参数量也有了较大的减少。值得一提的是,用1×1卷积核实现降维的思想,在后面的多个轻量化网络中都会使用到。

Inception v1网络一共有9个上述堆叠的模块,共有22层,在最后的Inception模块处使用了全局平均池化。为了避免深层网络训练时带来的梯度消
失问题,作者还引入了两个辅助的分类器,在第3个与第6个Inception模块输出后执行Softmax并计算损失,在训练时和最后的损失一并回传。
'''
# 下面使用PyTorch来搭建一个单独的Inception模块
import torch.nn.functional as F

# 首先定义一个包含conv和ReLU的基础卷积类
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
    def forward(self, x):
        x = self.conv(x)
        return F.relu(x, inplace=True)
# Inceptionv1的类，初始化时需要提供各个子模块的通道数大小
class Inceptionv1(nn.Module):
    def __init__(self, in_dim, hid_1_1, hid_2_1, hid_2_3, hid_3_1, out_3_5, out_4_1):
        super(Inceptionv1, self).__init__()
        # 下面分别是4个子模块各自的网络定义
        self.branch1x1 = BasicConv2d(in_dim, hid_1_1, 1)
        self.branch3x3 = nn.Sequential(
            BasicConv2d(in_dim, hid_2_1, 1),
            BasicConv2d(hid_2_1, hid_2_3, 3, padding=1)
        )
        self.branch5x5 = nn.Sequential(
            BasicConv2d(in_dim, hid_3_1, 1),
            BasicConv2d(hid_3_1, out_3_5, 5, padding=2)
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            BasicConv2d(in_dim, out_4_1, 1)
        )
    def forward(self, x):
        b1 = self.branch1x1(x)
        b2 = self.branch3x3(x)
        b3 = self.branch5x5(x)
        b4 = self.branch_pool(x)
        # 将这4个子模块沿着通道方向进行拼接
        output = torch.cat((b1, b2, b3, b4), dim=1)
        return output


if __name__ == "__main__":
    # 网络实例化，输入模块通道数，并转移到GPU上
    net_inceptionv1 = Inceptionv1(3, 64, 32, 64, 64, 96, 32).cuda()
    print('net_inceptionv1', net_inceptionv1)
    '''
    Inceptionv1(
      (branch1x1): BasicConv2d(
        (conv): Conv2d(3, 64, kernel_size=(1, 1), stride=(1, 1))
      )
      (branch3x3): Sequential(
        (0): BasicConv2d(
          (conv): Conv2d(3, 32, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): BasicConv2d(
          (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
      (branch5x5): Sequential(
        (0): BasicConv2d(
          (conv): Conv2d(3, 64, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): BasicConv2d(
          (conv): Conv2d(64, 96, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        )
      )
      (branch_pool): Sequential(
        (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
        (1): BasicConv2d(
          (conv): Conv2d(3, 32, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )
    '''
    input = torch.randn(1, 3, 256, 256).cuda()
    print('input.shape', input.shape)  # torch.Size([1, 3, 256, 256])
    output = net_inceptionv1(input)
    print('output.shape', output.shape)  # torch.Size([1, 256, 256, 256])
