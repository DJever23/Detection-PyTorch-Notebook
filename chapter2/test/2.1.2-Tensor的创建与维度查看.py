import torch

# 最基础的Tensor()函数创建方法，参数为Tensor的每一维大小
a = torch.Tensor(2,2)
print('a',a)
b = torch.DoubleTensor(2,2)
print('b',b)
# 使用Python的list序列进行创建
c = torch.Tensor([[1,2],[3,4]])
print('c',c)
# 使用zeros()函数，所有元素均为0
d = torch.zeros(2,2)
print('d',d)
# 使用ones()函数，所有元素均为1
e = torch.ones(2,2)
print('e',e)
# 使用eye()函数，对角线元素为1，不要求行列数相同，生成二维矩阵
f = torch.eye(4,7)
print('f',f)
# 使用randn()函数，生成随机数矩阵
g = torch.randn(2,2)
print('g',g)
# 使用arange(start, end, step)函数，表示从start到end，间距为step，一维向量
h = torch.arange(1,10,3)
print('h',h)  # tensor([1, 4, 7])
# 使用linspace(start, end, steps)函数，表示从start到end，一共steps份，一维向量
i = torch.linspace(1,6,4)  # tensor([1.0000, 2.6667, 4.3333, 6.0000])
print('i',i)
# 使用randperm(num)函数，生成长度为num的随机排列向量
j = torch.randperm(8)  # tensor([0, 1, 7, 2, 5, 4, 3, 6])
print('j',j)
# PyTorch 0.4中增加了torch.tensor()方法，参数可以为Python的list、NumPy的ndarray等
k = torch.tensor([1,2,3])
print('k',k)


'''
对于Tensor的维度，可使用Tensor.shape或者size()函数查看每
一维的大小，两者等价。
'''
A = torch.randn(2,2)
print('A.shape', A.shape)  # 使用shape查看Tensor维度
print('A.size', A.size())  # 使用size()函数查看Tensor维度

'''
查看Tensor中的元素总个数，可使用Tensor.numel()或者
Tensor.nelement()函数，两者等价。
'''
print('A.numel', A.numel())
print('A.nelement', A.nelement())
