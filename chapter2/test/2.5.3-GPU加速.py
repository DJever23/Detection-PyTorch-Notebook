'''
PyTorch为数据在GPU上运行提供了非常便利的操作。首先可以使用torch.cuda.is_available()来判断当前环境下GPU是否可用,其次
是对于Tensor和模型,可以直接调用cuda()方法将数据转移到GPU上运行,并且可以输入数字来指定具体转移到哪块GPU上运行。
'''
import torch
from torchvision import models
a = torch.randn(3, 3)
b = models.vgg16()
# 判断当前GPU是否可用
if torch.cuda.is_available():
    a = a.cuda()
    # 指定将b转移到编号为1的GPU上
    b = b.cuda(0)
# 使用torch.device()来指定使用哪一个GPU
device = torch.device("cuda: 0")
c = torch.randn(3, 3, device=device, requires_grad=True)

'''
对于在全局指定使用哪一块GPU,官方给出了两种方法,首先是在终端执行脚本时直接指定GPU的方式:
CUDA_VISIBLE_DEVICES=2 python3 train.py
其次是在脚本中利用函数指定,如下:
import torch
torch.cuda.set_device(1)
官方建议使用第一种方法,即CUDA_VISIBLE_DEVICE的方式。
'''

# 在工程应用中,通常使用torch.nn.DataParallel(module,device_ids)函数来处理多GPU并行计算的问题。
from torch import nn
model_gpu = nn.DataParallel(model, device_ids=[0, 1])
output = model_gpu(input)
'''
多GPU处理的实现方式是,首先将模型加载到主GPU上,然后复制模型到各个指定的GPU上,将输入数据按batch的维度进行划分,分配
到每个GPU上独立进行前向计算,再将得到的损失求和并反向传播更新单个GPU上的参数,最后将更新后的参数复制到各个GPU上。
'''
