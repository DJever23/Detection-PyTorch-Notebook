import torch

'''
nn.Module是PyTorch提供的神经网络类，并在类中实现了网络各层的定
义及前向计算与反向传播机制。在实际使用时，如果想要实现某个神经网
络，只需继承nn.Module，在初始化中定义模型结构与参数，在函数
forward()中编写网络前向过程即可。
下面具体以一个由两个全连接层组成的感知机为例，介绍如何使用
nn.Module构造模块化的神经网络。
'''
from torch import nn
import torch.nn.functional as F
'''
在深度学习中,损失反映模型最后预测结果与实际真值之间的差距,可以用来分析训练过程的好坏、模型是否收敛等,例如均方损
失、交叉熵损失等。
利用nn.functional定义的网络层不可自动学习参数,还需要使用nn.Parameter封装。nn.functional的设计初
衷是对于一些不需要学习参数的层,如激活层、BN(Batch Normalization)层,可以使用nn.functional,这样这些层就不需要在
nn.Module中定义了。
'''


# 首先建立一个全连接的子module，继承nn.Module
class Linear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Linear, self).__init__()  # 调用nn.Module的构造函数
        # 使用nn.Parameter来构造需要学习的参数
        self.w = nn.Parameter(torch.randn(in_dim, out_dim))
        self.b = nn.Parameter(torch.randn(out_dim))

    # 在forward中实现前向传播过程
    def forward(self, x):
        x = x.matmul(self.w)  # 使用Tensor.matmul实现矩阵相乘
        y = x + self.b.expand_as(x)  # 使用Tensor.expand_as()来保证矩阵形状一致
        return y
    # 构建感知机类，继承nn.Module,并调用了Linear的子module


class Perception(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(Perception, self).__init__()
        self.layer1 = Linear(in_dim, hid_dim)
        self.layer2 = Linear(hid_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        y = torch.sigmoid(x)  # 使用torch中的sigmoid作为激活函数
        y = self.layer2(y)
        y = torch.sigmoid(y)
        return y


if __name__ == '__main__':
    # 实例化一个网络，并赋值全连接中的维数，最终输出二维代表了二分类
    perception = Perception(2, 3, 2)
    # 可以看到perception中包含上述定义的layer1与layer2
    print('perception', perception)
    # named_parameters()可以返回学习参数的迭代器，分别为参数名与参数值
    for name, parameters in perception.named_parameters():
        print(name, parameters)
    #随机生成数据，注意这里的4代表了样本数为4，每个样本有两维
    data = torch.randn(2, 2)
    print('data', data)
    # 将输入数据传入perception，perception()相当于调用perception中的forward()函数
    output = perception(data)
    print('output', output)  # 输出的两个维度代表属于两个类别的sigmoid值

    # 设置标签，由于是二分类，一共有4个样本，因此标签维度为4，每个数为0或1两个类别
    label = torch.Tensor([0, 1]).long()
    # 实例化nn的交叉熵损失类
    criterion = nn.CrossEntropyLoss()
    # 调用交叉熵损失
    loss_nn = criterion(output, label)
    print('loss_nn', loss_nn)
    # 由于F.cross_entropy是一个函数，因此可以直接调用，不需要实例化，两者求得的损失值相同
    '''
    PyTorch在torch.nn及torch.nn.functional中都提供了各种损失函数,通常来讲,由于损失函数不含有可学习的参数,因此这两者在
    功能上基本没有区别。
    '''
    loss_functional = F.cross_entropy(output, label)
    print('loss_functional', loss_functional)