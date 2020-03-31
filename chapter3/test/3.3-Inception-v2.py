'''
在Inception v1网络的基础上,随后又出现了多个Inception版本。Inception v2进一步通过卷积分解与正则化实现更高效的计算,增加了BN层,同
时利用两个级联的3×3卷积取代了Inception v1版本中的5×5卷积,这种方式既减少了卷积参数量,也增加了网络的非线性能力。
'''
# 使用PyTorch来搭建一个单独的Inception v2模块,默认输入的通道数为192
import torch
from torch import nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    # 构建基础的卷积模块，与v1的基础模块相比，增加了BN层
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionV2(nn.Module):
    def __init__(self):
        super(InceptionV2, self).__init__()
        self.branch1 = BasicConv2d(192, 96, 1, 0)  # 对应1x1卷积分支
        self.branch2 = nn.Sequential(
            BasicConv2d(192, 48, 1, 0),
            BasicConv2d(48, 64, 3, 1)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(192, 64, 1, 0),
            BasicConv2d(64, 96, 3, 1),
            BasicConv2d(96, 96, 3, 1)
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(192, 64, 1, 0)
        )
        # 前向过程，将4个分支进行torch.cat()拼接起来
    def forward(self, x):
        x0 = self.branch1(x)
        x1 = self.branch2(x)
        x2 = self.branch3(x)
        x3 = self.branch4(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


if __name__ == "__main__":
    net_inceptionv2 = InceptionV2().cuda()
    print('net_inceptionv2', net_inceptionv2)
    '''
    InceptionV2(
      (branch1): BasicConv2d(
        (conv): Conv2d(192, 96, kernel_size=(1, 1), stride=(1, 1))
        (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
      )
      (branch2): Sequential(
        (0): BasicConv2d(
          (conv): Conv2d(192, 48, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): BasicConv2d(
          (conv): Conv2d(48, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (branch3): Sequential(
        (0): BasicConv2d(
          (conv): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): BasicConv2d(
          (conv): Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
        (2): BasicConv2d(
          (conv): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (bn): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (branch4): Sequential(
        (0): AvgPool2d(kernel_size=3, stride=1, padding=1)
        (1): BasicConv2d(
          (conv): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
    )
    '''
    input = torch.randn(1, 192, 32, 32).cuda()
    print('input.shape', input.shape)  # torch.Size([1, 192, 32, 32])
    output = net_inceptionv2(input)
    print('output.shape', output.shape)  # torch.Size([1, 320, 32, 32])

'''
更进一步,Inception v2将n×n的卷积运算分解为1×n与n×1两个卷积,这种分解的方式可以使计算成本降低33%。
此外,Inception v2还将模块中的卷积核变得更宽而不是更深,形成第三个模块,以解决表征能力瓶颈的问题。
Inception v2网络正是由上述的三种不同类型的模块组成的,其计算也更加高效。
Inception v3在Inception v2的基础上,使用了RMSProp优化器,在辅助的分类器部分增加了7×7的卷积,并且使用了标签平滑技术。
Inception v4则是将Inception的思想与残差网络进行了结合,显著提升了训练速度与模型准确率.
'''