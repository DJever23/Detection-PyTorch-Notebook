import torch

'''
PyTorch在0.2版本以后，推出了自动广播语义，即不同形状的
Tensor进行计算时，可自动扩展到较大的相同形状，再进行计算。广播
机制的前提是任一个Tensor至少有一个维度，且从尾部遍历Tensor维度
时，两者维度必须相等，其中一个要么是1要么不存在。
'''
a = torch.ones(3, 1, 2)
print('a', a)
b = torch.ones(2, 1)
print('b', b)
# 从尾部遍历维度，1对应2，2对应1，3对应不存在，因此满足广播条件，最后求和后的维度为[3,2,2]???
print('(a+b).size()', (a+b).size())
print('a+b', a+b)
c = torch.ones(1, 2)
print('c', c)
print('(a+c).size()', (a+c).size())
print('a+c', a+c)