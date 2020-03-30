import torch
'''
下面利用PyTorch来搭建常用的优化器,传入参数包括网络中需要
学习优化的Tensor对象、学习率和权值衰减等。
'''
from torch import optim
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim1, hid_dim2, out_dim):
        super(MLP, self).__init__()
        # 通过Sequential快速搭建三层感知机
        self.layer = nn.Sequential(
            nn.Linear(in_dim,hid_dim1),
            nn.ReLU(),
            nn.Linear(hid_dim1, hid_dim2),
            nn.ReLU(),
            nn.Linear(hid_dim2, out_dim),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.layer(x)
        return x

if __name__ == "__main__":
    # 实例化模型，并赋予每一层的维度
    model = MLP(28*28, 300, 200, 10)
    print(model)  # 打印model的结构，由3个全连接层组成
    #采用SGD优化器，学习率为0.01
    optimizer1 = optim.SGD(params = model.parameters(), lr=0.01)
    '''
    不同参数层分配不同的学习率:优化器也可以很方便地实现将
    不同的网络层分配成不同的学习率,即对于特殊的层单独赋予学习
    率,其余的保持默认的整体学习率,具体示例如下:
    '''
    # 对于model中需要单独赋予学习率的层,如special层,则使用'lr'关键字单独赋予
    # optimizer1 = optim.SGD({'params': model.special.parameters(), 'lr': 0.001},{'params': model.base.parameters()}, lr=0.0001)
    data = torch.randn(10, 28*28)
    output = model(data)
    # 由于是10分类，因此label元素从0到9，一共10个样本
    label = torch.Tensor([1, 0, 4, 7, 9, 3, 4, 5, 3, 2]).long()
    print(label)
    # 求损失
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, label)
    print('loss', loss)
    # 清空梯度，在每次优化前都需要进行此操作
    optimizer1.zero_grad()
    # 损失的反向传播
    loss.backward()
    # 利用优化器进行梯度更新
    optimizer1.step()