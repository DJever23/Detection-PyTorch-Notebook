'''
ResNet的思想在于引入了一个深度残差框架来解决梯度消失问题,即让卷积网络去学习残差映射,而不是期望每一个堆叠层的网络都完整
地拟合潜在的映射(拟合函数)。
ResNet通过前层与后层的“短路连接”(Shortcuts),加强了前后层之间的信息流通,在一定程度上缓解了梯度消失现象,从而可以将神经网络搭建得很深。
利用PyTorch实现一个带有Downsample操作的Bottleneck结构
'''
import torch
from torch import nn


class Bottleneck(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1):
        super(Bottleneck, self).__init__()
        # 网络堆叠层是由1x1、 3x3、 1x1这3个卷积组成的，中间包含BN层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim, 3, stride, 1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim)
        )
        self.relu = nn.ReLU(inplace=True)
        # Downsample部分是由一个包含BN的1x1卷积组成
        self.downsample = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, 1),
            nn.BatchNorm2d(out_dim)
        )

    def forward(self, x):
        # identity = x
        out = self.bottleneck(x)
        identity = self.downsample(x)
        # 将identity(恒等映射)与网络堆叠层输出进行相加，并经过ReLU后输出
        out += identity
        out = self.relu(out)
        return out


if __name__ == '__main__':
    # 实例化Bottleneck，输入通道数为64，输出为256，对应第一个卷积组的第一个Bottleneck
    bottleneck_1_1 = Bottleneck(64, 256).cuda()
    print('bottleneck_1_1', bottleneck_1_1)
    # Bottleneck作为卷积堆叠层，包含了1x1,3x3,1x1这3个卷积层
    '''
    Bottleneck(
      (bottleneck): Sequential(
        (0): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
        (6): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (relu): ReLU(inplace=True)
      # 利用Downsample结构将恒等映射的通道数变为与卷积堆叠层相同,保证可以相加
      (downsample): Sequential(
        (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    '''
    input = torch.randn(1, 64, 56, 56).cuda()
    output = bottleneck_1_1(input)
    print('input.shape', input.shape)  # torch.Size([1, 64, 56, 56])
    print('output.shape', output.shape)
    # torch.Size([1, 256, 56, 56]),相比输入，输出的特征图分辨率没变，而通道数变为4倍