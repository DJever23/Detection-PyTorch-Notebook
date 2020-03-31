import torch
from torch import nn
'''
在VGGNet中,使用的卷积核基本都是3×3,而且很多地方出现了多个3×3堆叠的现象,这种结构的优点在于,
首先从感受野来看,两个3×3的卷积核与一个5×5的卷积核是一样的;其次,同等感受野时,3×3卷积核的参数量更少。更为重要的是,两个3×3卷
积核的非线性能力要比5×5卷积核强,因为其拥有两个激活函数,可大大提高卷积网络的学习能力。
'''
# 下面使用PyTorch来搭建VGG16经典网络结构
class VGG(nn.Module):
    def __init__(self,num_classes=1000):
        super(VGG,self).__init__()
        layers = []
        in_dim = 3
        out_dim =64
        # 循环构造卷积层，一共有13个卷积层
        for i in range(13):
            layers += [nn.Conv2d(in_dim, out_dim, 3, 1, 1), nn.ReLU(inplace=True)]
            in_dim = out_dim
            # 在第2\4\7\10\13个卷积后增加池化层
            if i==1 or i==3 or i==6 or i==9 or i==12:
                layers += [nn.MaxPool2d(2, 2)]
                # 第10个卷积后保持和前边的通道数一致，都为512，其余加倍
                if i!=9:
                    out_dim *= 2
        self.features = nn.Sequential(*layers)  # 这里加*的含义是什么？
        # VGGNet的3个全连接层，中间有ReLU和Dropout层
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        # 这里是将特征图的维度从[1,512,7,7]变到[1,512*7*7]
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # 实例化VGG类，在此设置输出分类数为21，并转移到GPU上
    vgg = VGG(num_classes=21).cuda()
    input = torch.randn(1, 3, 224, 224).cuda()
    print('input shape', input.shape)  # torch.Size([1, 3, 224, 224])
    # 调用VGG，输出21类的得分
    scores = vgg(input)
    print('scores', scores)
    '''
    tensor([[-0.0034,  0.0109,  0.0108, -0.0066, -0.0150,  0.0088,  0.0066,  0.0055,
          0.0016, -0.0048, -0.0041, -0.0064, -0.0179, -0.0204, -0.0046,  0.0116,
         -0.0016,  0.0058,  0.0107, -0.0043,  0.0054]], device='cuda:0',
       grad_fn=<AddmmBackward>)
    '''
    print('scores.shape', scores.shape)  # torch.Size([1, 21])
    # 也可以单独调用卷积模块，输出最后一层的特征图
    features = vgg.features(input)
    print('features.shape', features.shape)  # torch.Size([1, 512, 7, 7])
    # 打印出VGGNet的卷积层，5个卷积组一共31层
    print('vgg.features', vgg.features)
    '''
    Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    '''
    # 打印VGGNet的3个全连接层
    print('vgg.classifier', vgg.classifier)
    '''
    Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=21, bias=True)
    )
    '''