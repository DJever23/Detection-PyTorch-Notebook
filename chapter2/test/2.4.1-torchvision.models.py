'''
对于深度学习,torchvision.models库提供了众多经典的网络结
构与预训练模型,例如VGG、ResNet和Inception等,利用这些模型可
以快速搭建物体检测网络,不需要逐层手动实现.

以VGG模型为例,在torchvision.models中,VGG模型的特征层与
分类层分别用vgg.features与vgg.classifier来表示,每个部分是一
个nn.Sequential结构,可以方便地使用与修改。下面讲解如何使用
torchvision.model模块。
'''
from torch import nn
from torchvision import models
# 通过torchvision.model直接调用VGG16网络结构
vgg = models.vgg16()
# VGG16的特征层包括13个卷积、13个激活函数ReLU、5个池化，一共31层
print(len(vgg.features))
# VGG16的分类层3个全连接、2个ReLU、2个Dropout，一共7层
print(len(vgg.classifier))
# 可以通过出现的顺序直接索引每一层
print(vgg.classifier[-1])
# 也可以选取某一部分，如下代表了特征网络的最后一个卷积模组
print(vgg.features[24:])