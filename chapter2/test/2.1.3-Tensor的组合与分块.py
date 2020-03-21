import torch

'''
组合操作是指将不同的Tensor叠加起来，主要有torch.cat()和
torch.stack()两个函数。cat即concatenate的意思，是指沿着已有的
数据的某一维度进行拼接，操作后数据的总维数不变，在进行拼接
时，除了拼接的维度之外，其他维度必须相同。而torch.stack()函数
指新增维度，并按照指定的维度进行叠加.
'''

# 创建两个2×2的Tensor
a = torch.Tensor([[1, 2], [3, 4]])
b = torch.Tensor([[5, 6], [7, 8]])
print(torch.full_like(a,3))
print('a', a)
print('b', b)
# 以第一维进行拼接，则变成4×2的矩阵
c = torch.cat([a, b], 0)
print('c', c)
# 以第二维进行拼接，则变成2x4的矩阵
d = torch.cat([a, b], 1)
print('d', d)
# 以第0维进行stack，叠加的基本单位为序列本身，即a与b，因此输出[a, b]，输出维度为2×2×2
e = torch.stack([a, b], 0)
print('e', e)
# 以第1维进行stack，叠加的基本单位为每一行，输出维度为2×2×2
f = torch.stack([a, b], 1)
print('f', f)
# 以第2维进行stack，叠加的基本单位为每一行的每一个元素，输出维度为2×2×2
g = torch.stack([a, b], 2)
print('g', g)


'''
分块则是与组合相反的操作，指将Tensor分割成不同的子
Tensor，主要有torch.chunk()与torch.split()两个函数，前者需要
指定分块的数量，而后者则需要指定每一块的大小，以整型或者list
来表示。
'''
A = torch.Tensor([[1, 2, 3], [4, 5, 6]])
print('A', A)
# 使用chunk，沿着第0维进行分块，一共分两块，因此分割成两个1×3的Tensor
B = torch.chunk(A, 2, 0)
print('B', B)
# 沿着第1维进行分块，因此分割成两个Tensor，当不能整除时，最后一个的维数会小于前面的
# 因此第一个Tensor为2×2，第二个为2×1
C = torch.chunk(A, 2, 1)
print('C', C)
# 使用split，沿着第0维分块，每一块维度为2，由于第一维维度总共为2，因此相当于没有分割
D = torch.split(A, 2, 0)
print('D', D)
# 沿着第1维分块，每一块维度为2，因此第一个Tensor为2×2，第二个为2×1
E = torch.split(A, 2, 1)
print('E', E)
# split也可以根据输入的list进行自动分块，list中的元素代表了每一个块占的维度
F = torch.split(A, [1, 2], 1)
print('F', F)

