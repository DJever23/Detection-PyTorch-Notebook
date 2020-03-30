'''
空洞卷积的优点显而易见,在不引入额外参数的前提下可以任意扩大感受野,同时保持特征图的分辨率不变。这一点在分割与检测任务中十分有
用,感受野的扩大可以检测大物体,而特征图分辨率不变使得物体定位更加精准。
PyTorch对于空洞卷积也提供了方便的实现接口,在卷积时传入dilation参数即可。
'''
from torch import nn
# 定义普通卷积，默认dilation为1
conv1 = nn.Conv2d(3, 256, 3, stride=1, padding=1, dilation=1)
print('conv1', conv1)
# 定义dilation为2的卷积，打印卷积后会有dilation的参数
conv2 = nn.Conv2d(3, 256, 3, stride=1, padding=1, dilation=2)
print('conv2', conv2)
'''
空洞卷积的缺点：
1.网格效应(Gridding Effect)
2.远距离的信息没有相关性
3.不同尺度物体的关系:大的dilation rate对于大物体分割与检测有利,但是对于小物体则有弊无利,如何处理好多尺度问题的检测,是空洞卷
积设计的重点。
对于上述问题,有多篇文章提出了不同的解决方法,典型的有图森未来提出的HDC(Hybrid Dilated Convolution)结构。
该结构的设计准则是堆叠卷积的dilation rate不能有大于1的公约数,同时将dilation rate设置为类似于[1,2,5,1,2,5]
这样的锯齿类结构。此外各dilation rate之间还需要满足一个数学公式,这样可以尽可能地覆盖所有空洞,以解决网格效应与远距
离信息的相关性问题,具体细节可参考相关资料。
'''