import torch

'''
索引操作与NumPy非常类似，主要包含下标索引、表达式索引、使
用torch.where()与Tensor.clamp()的选择性索引。
'''
a = torch.Tensor([[0, 1], [2, 3]])
print('a', a)
# 根据下标进行索引
print('a[1]', a[1])
print('a[0, 1]', a[0, 1])
# 选择a中大于0的元素，返回和a相同大小的Tensor，符合条件的置1，否则置0
print('a>0', a>0)
# 选择符合条件的元素并返回，等价于torch.masked_select(a, a>0)
print('a[a>0]',a[a>0])
# 选择非0元素的坐标，并返回
print('torch.nonzero(a)', torch.nonzero(a))
# torch.where(condition, x, y)，满足condition的位置输出x，否则输出y
print('torch.where(a>1, torch.full_like(a,1),a)', torch.where(a > 2, torch.full_like(a, 5), a))#([0,1],[2,5])
# 对Tensor元素进行限制可以使用clamp()函数，示例如下，限制最小值为1，最大值为2
print('a.clamp(1,2)', a.clamp(1,2))

'''
变形操作则是指改变Tensor的维度，以适应在深度学习的计算
中，数据维度经常变换的需求，是一种十分重要的操作。在PyTorch中
主要有4类不同的变形方法
'''
# 1．view()、resize()和reshape()函数:可以在不改变Tensor数据的前
# 提下任意改变Tensor的形状，必须保证调整前后的元素总数相同，并
# 且调整前后共享内存，三者的作用基本相同。
print('1．view()、resize()和reshape()函数')
b = torch.arange(1,5)
print('b', b)
# 分别使用view()、resize()及reshape()函数进行维度变换
c = b.view(2, 2)
print('c', c)
#d = b.resize(4, 1)  # resize has warning
#print('d', d)
e = b.reshape(4, 1)
print('e', e)
# 如果想要直接改变Tensor的尺寸，可以使用resize_()的原地操作函数。
f = b.resize_(2, 3)
print('f', f)
# 改变了b、c、d的一个元素，a也跟着改变了，说明两者共享内存
c[0, 0] = 0
#d[1, 0] = 0
print('b', b)


# 2．transpose()和permute()函数,transpose()函数可以将指定的两个维度的元素进行转置，而
# permute()函数则可以按照给定的维度进行维度变换。
print('2．transpose()和permute()函数')
A = torch.randn(2, 2, 3)
print('A', A)
# 将第0维和第1维的元素进行转置
print('A.transpose(0,1)', A.transpose(0, 1))
# 按照第2、1、0的维度顺序重新进行元素排列
print('A.permute(2,1,0)', A.permute(2, 1, 0))

'''
3．squeeze()和unsqueeze()函数
在实际的应用中，经常需要增加或减少Tensor的维度，尤其是维
度为1的情况，这时候可以使用squeeze()与unsqueeze()函数，前者用
于去除size为1的维度，而后者则是将指定的维度的size变为1。
'''
print('3．squeeze()和unsqueeze()函数')
a = torch.arange(1, 4)
print('a.shape', a.shape)
# 将第0维变为1，因此总的维度为1、3
print('a.unsqueeze(0).shape', a.unsqueeze(0).shape)
# 第0维如果是1，则去掉该维度，如果不是1则不起任何作用
print('a.unsqueeze(0).squeeze(0).shape', a.unsqueeze(0).squeeze(0).shape)


'''
4．expand()和expand_as()函数
有时需要采用复制元素的形式来扩展Tensor的维度，这时expand
就派上用场了。expand()函数将size为1的维度复制扩展为指定大小，
也可以使用expand_as()函数指定为示例Tensor的维度。
'''
print('4．expand()和expand_as()函数')
a = torch.randn(2, 2, 1)
print('a',a)
# 将第2维的维度由1变为3，则复制该维的元素，并扩展为3
print('a.expand(2,2,3)', a.expand(2, 2, 3))


