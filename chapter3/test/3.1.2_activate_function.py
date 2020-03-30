import torch
from torch import nn
# 1.Sigmoid函数, Sigmoid型函数又称为Logistic函数, 可以用来做二分类,但其计算量较大,并且容易出现梯度消失现象
input1 = torch.ones(1, 1, 2, 2)
print('input1', input1)
sigmoid = nn.Sigmoid()
print(sigmoid(input1))

# 2.ReLu函数
'''
为了缓解梯度消失现象,修正线性单元(Rectified Linear Unit,ReLU)被引入到神经网络中。由于其优越的性能与简单优雅的实现,
ReLU已经成为目前卷积神经网络中最为常用的激活函数之一。
'''
input2 = torch.randn(1, 1, 2, 2)
print('input2', input2)
# nn.ReLU()可以实现inplace操作，即可以直接将运算结果覆盖到输入中，以节省内存，即input的值被修改
relu = nn.ReLU(inplace=False)
print('relu', relu(input2))

# 3.Leaky ReLU函数
'''
ReLU激活函数虽然高效,但是其将负区间所有的输入都强行置为0,Leaky ReLU函数优化了这一点,在负区间内避免了直接置0,而是赋予很
小的权重.虽然从理论上讲,Leaky ReLU函数的使用效果应该要比ReLU函数好,但是从大量实验结果来看并没有看出其效果比ReLU好。此外,对于
ReLU函数的变种,除了Leaky ReLU函数之外,还有PReLU和RReLU函数等。
'''
# 利用nn.LeakyReLU()构建激活函数，并且权重为0.04，True代表inplace操作
leakyrelu = nn.LeakyReLU(0.04, True)
print('leaktrelu', leakyrelu(input2))

# 4.Softmax函数
'''
在具体的分类任务中,Softmax函数的输入往往是多个类别的得分,输出则是每一个类别对应的概率,所有类别的概率取值都在0~1之间,且
和为1.在PyTorch中,Softmax函数在torch.nn.functional库中
'''
import torch.nn.functional as F
score = torch.randn(2, 4)
print('score', score)
# 利用torch.nn.functional.softmax()函数，第二个参数表示按照第几个维度进行Softmax计算
print(F.softmax(score, 0))