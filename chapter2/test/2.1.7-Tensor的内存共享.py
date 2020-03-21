import torch

'''
1．通过Tensor初始化Tensor
直接通过Tensor来初始化另一个Tensor，或者通过Tensor的组
合、分块、索引、变形操作来初始化另一个Tensor，则这两个Tensor
共享内存.
'''
print('1．通过Tensor初始化Tensor')
a = torch.randn(2, 2)
print('a', a)
# 用a初始化b，或者用a的变形操作初始化c，这三者共享内存，一个变，其余的也改变了
b = a
c = a.view(4)
b[0, 0] = 0
c[3] = 4
print('a_1', a)

'''
2．原地操作符
PyTorch对于一些操作通过加后缀“_”实现了原地操作，如
add_()和resize_()等，这种操作只要被执行，本身的Tensor则会被改
变。
'''
print('2．原地操作符')
a = torch.Tensor([[1, 2], [3, 4]])
# add_()函数使得a也改变了
b = a.add_(a)
print('a.add_', a)
# resize_()函数使得a也发生了改变
c = a.resize_(4)
print('a_resize_', a)
print('c', c)

'''
3．Tensor与NumPy转换
Tensor与NumPy可以高效地进行转换，并且转换前后的变量共享内
存。在进行PyTorch不支持的操作时，甚至可以曲线救国，将Tensor转
换为NumPy类型，操作后再转为Tensor。
'''
print('3．Tensor与NumPy转换')
a = torch.randn(1, 2)
print('a', a)
# Tensor转为NumPy
b = a.numpy()
print('b', b)
# NumPy转为Tensor
c = torch.from_numpy(b)
print('c', c)
#Tensor转为list
d = a.tolist()
print('d', d)