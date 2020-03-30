'''
对于计算机视觉的任务,包括物体检测,我们通常很难拿到很大的数据集,在这种情况下重新训练一个新的模型是比较复杂的,并且
不容易调整,因此,Fine-tune(微调)是一个常用的选择。所谓Fine-tune是指利用别人在一些数据集上训练好的预训练模型,在自己
的数据集上训练自己的模型。
'''
import torch
from torch import nn
from torchvision import models
# 第一种是直接利用torchvision.models中自带的与训练模型，在使用时赋予pretrained参数为True即可
# 通过torchvision.models直接调用VGG16网络结构
vgg = models.vgg16(pretrained=True)


# 第二种是如果想要使用自己的本地预训练模型,或者之前训练过的模型,则可以通过model.load_state_dict()函数操作
# 通过torchvision.models直接调用VGG16网络结构
vgg1 = models.vgg16()
state_dict = torch.load("your model path")
# 利用load_state_dict,遍历预训练模型的关键字，如果出现在了VGG中，则加载预训练参数
vgg1.load_state_dict({k:v for k,v in state_dict.items() if k in vgg1.state_dict()})


'''
通常来讲,对于不同的检测任务,卷积网络的前两三层的作用是非常类似的,都是提取图像的边缘信息等,因此为了保证模型训练中
能够更加稳定,一般会固定预训练网络的前两三个卷积层而不进行参数的学习。例如VGG模型,可以设置前三个卷积模组不进行参数学习,
设置方式如下:
'''
for layer in range(10):
    for p in vgg1[layer].parameters():
        p.requires_grad = False

'''
在PyTorch中,参数的保存通过torch.save()函数实现,可保存对象包括网络模型、优化器等,而这些对象的当前状态数据可以通过自
身的state_dict()函数获取。
'''
'''
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_path.pth'
    })
'''
