import torch
from torch import nn
'''
Dropout算法,可以比较有效地缓解过拟合现象的发生,起到一定正则化的效果。
在训练时,每个神经元以概率p保留,即以1-p的概率停止工作,每次前向传播保留下来的神经元都不同,
这样可以使得模型不太依赖于某些局部特征,泛化性能更强。
Dropout被广泛应用到全连接层中,一般保留概率设置为0.5,而在较为稀疏的卷积网络中则一般使用下一节将要介绍的BN层来正则化
模型,使得训练更稳定。
'''
# PyTorch将元素置0来实现Dropout层，第一个参数为置零概率，第二个为是否原地操作
dropout = nn.Dropout(0.5, inplace=False)
input = torch.randn(2, 64, 7, 7)
output = dropout(input)
