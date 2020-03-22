import torch

'''
计算图是PyTorch对于神经网络的具体实现形式，包括每一个数据Tensor及Tensor之间的函数function。
Autograd的基本原理是随着每一步Tensor的计算操作，逐渐生成计算图，并将操作的function记录在
Tensor的grad_fn中。在前向计算完后，只需对根节点进行backward函数操作，即可从当前根节点自动
进行反向传播与梯度计算，从而得到每一个叶子节点的梯度，梯度计算遵循链式求导法则。
'''
# 生成3个Tensor变量，并作为叶节点
x = torch.randn(1)
print(x)  # tensor([0.6114])
w = torch.ones(1, requires_grad=True)
b = torch.ones(1, requires_grad=True)
# 自己生成的，因此都为叶节点
print(x.is_leaf, w.is_leaf, b.is_leaf)  # True True True
# 默认是不需要求导，关键字赋值为True后则需要求导
print(x.requires_grad, w.requires_grad, b.requires_grad)  # False True True
# 进行前向计算，由计算生成的变量都不是叶节点
y = w * x
z = y + b
print(y.is_leaf, z.is_leaf)  # False False
# 由于依赖的变量有需要求导的，因此y与z都需要求导
print(y.requires_grad, z.requires_grad)  # True True
# grad_fn记录生成该变量经过了什么操作，如y是Mul，z是Add
print(y.grad_fn)  # <MulBackward0 object at 0x7f52479ac518>
print(z.grad_fn)  # <AddBackward0 object at 0x7f52479ac518>
# 对根节点调用backward()函数，进行梯度反传
z.backward(retain_graph = True)
print(w.grad)  # tensor([0.6114])
print(b.grad)  # tensor([1.])

'''
Autograd注意事项
1.动态图特性:PyTorch建立的计算图是动态的.
2.backward()函数还有一个需要传入的参数grad_variabels，其代表了根节点的导数，也可以看做根节点各部分的权重系数。
3.当有多个输出需要同时进行梯度反传时，需要将retain_graph设置为True，从而保证在计算多个输出的梯度时互不影响。
'''