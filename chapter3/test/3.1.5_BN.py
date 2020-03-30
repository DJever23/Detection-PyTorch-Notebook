from torch import nn
import torch
# 使用BN层需要传入一个参数为num_features,即特征的通道数
bn = nn.BatchNorm2d(64)
print('bn', bn)
input = torch.randn(4, 64, 224, 224)
output = bn(input)
# BN层不改变输入、输出的特征大小
# print('output', output)
print(output.shape)
'''
BN的优点：
缓解梯度消失,加速网络收敛。简化调参,网络更稳定。防止过拟合。
缺点：
1.由于是在batch的维度进行归一化,BN层要求较大的batch才能有效地工作,而物体检测等任务由于占用内存较高,限制了batch的大小,这会
限制BN层有效地发挥归一化功能。
2.数据的batch大小在训练与测试时往往不一样。在训练时一般采用滑动来计算平均值与方差,在测试时直接拿训练集的平均值与方差来使用。
这种方式会导致测试集依赖于训练集,然而有时训练集与测试集的数据分布并不一致。

因此,我们能不能避开batch来进行归一化呢?答案是可以的,最新的工作GN(Group Normalization)从通道方向计算均值与方差,使用更为灵
活有效,避开了batch大小对归一化的影响。GN先将特征图的通道分为很多个组,对每一个组内的参数做归一化,而不是batch。
'''