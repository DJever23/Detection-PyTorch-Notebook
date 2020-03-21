import torch
# 创建新Tensor，默认类型为torch.FloatTensor
a = torch.Tensor(2,2)#random
print('a',a)
# 使用int()、float()、double()等直接进行数据类型转换
b = a.int()
print('b',b)

c = a.type(torch.DoubleTensor)
print('c',c)

d = a.type_as(b)
print('d',d)