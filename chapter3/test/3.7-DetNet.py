import torch
from torch import nn


# 实现DetNet的两个Bottleneck结构A和B
class DetBottleneck(nn.Module):
    def __init__(self, inplanes, planes, srtide=1, extra=False):
        super(DetBottleneck, self).__init__()
        # 构建连续3个卷积层的Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.extra = extra
        # Bottleneck B的1x1卷积
        if self.extra:
            self.extra_conv = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        # 对于Bottleneck B来讲，需要对恒等映射增加卷积处理，与ResNet类似
        if self.extra:
            identity = self.extra_conv(x)
        else:
            identity = x
        out = self.bottleneck(x)
        out += identity
        out = self.relu(out)
        return out


if __name__ == '__main__':
    # 完成一个stage5，即B-A-A结构，stage4输出通道为1024
    bottleneck_b = DetBottleneck(1024, 256, 1, True).cuda()
    print('bottleneck_b', bottleneck_b)
    '''
    DetBottleneck(
      (bottleneck): Sequential(
        (0): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
        (6): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (relu): ReLU(inplace=True)
      (extra_conv): Sequential(
        (0): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    '''
    bottleneck_a1 = DetBottleneck(256, 256).cuda()
    print('bottleneck_a1', bottleneck_a1)
    '''
    DetBottleneck(
      (bottleneck): Sequential(
        (0): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
        (6): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (relu): ReLU(inplace=True)
    )
    '''
    bottleneck_a2 = DetBottleneck(256, 256).cuda()
    print('bottleneck_a2', bottleneck_a2)
    input = torch.randn(1, 1024, 14, 14).cuda()
    output1 = bottleneck_b(input)
    output2 = bottleneck_a1(output1)
    output3 = bottleneck_a1(output2)
    # 3个Bottleneck输出的特征图大小完全相同
    print(output1.shape, output2.shape, output3.shape)
    # torch.Size([1, 256, 14, 14]) torch.Size([1, 256, 14, 14]) torch.Size([1, 256, 14, 14])

