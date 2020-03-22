import torch

'''
自动求导机制记录了Tensor的操作，以便自动求导与反向传播。
可以通过requires_grad参数来创建支持自动求导机制的Tensor。

Tensor有两个重要的属性，分别记录了该Tensor的梯度与经历的操作。
    ·grad：该Tensor对应的梯度，类型为Tensor，并与Tensor同维
            度。
    ·grad_fn：指向function对象，即该Tensor经过了什么样的操
            作，用作反向传播的梯度计算，如果该Tensor由用户自己创建，
            则该grad_fn为None。
'''
a = torch.randn(2, 2, requires_grad=True)
b = torch.randn(2, 2)
# 可以看到默认的Tensor是不需要求导的，设置requires_grad为True后则需要求导
print(a.requires_grad, b.requires_grad)  # True False
# 也可以通过内置函数requires_grad_()将Tensor变为需要求导
print(b.requires_grad_())  # tensor([[ 0.2277, -1.1367],
                            # [-0.7707, -1.7638]], requires_grad=True)
print(b.requires_grad)  # True
# 通过计算生成的Tensor，由于依赖的Tensor需要求导，因此c也需要求导
c = a + b
print(c.requires_grad)  # True
# a与b是自己创建的，grad_fn为None，而c的grad_fn则是一个Add函数操作
print(a.grad_fn, b.grad_fn, c.grad_fn)  # None None <AddBackward0 object at 0x7f9ad9f5b7f0>
d = c.detach()  # Tensor.detach()函数生成的数据默认requires_grad为False。
print(d.requires_grad)  # False
