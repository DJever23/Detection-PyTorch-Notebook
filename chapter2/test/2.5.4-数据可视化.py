# 在PyTorch中,常用的可视化工具有TensorBoardX和Visdom。
'''
1.TensorBoardX是专为PyTorch开发的一套数据可视化工具,功能与TensorBoard相当,支持曲线、图片、文本和计算图等不同形式的可视化,而
且使用简单。下面以实现损失曲线的可视化为例,介绍TensorBoardX的使用方法。
安装TensorBoardX：
    pip install tensorboardX
'''
# 在训练脚本中,添加如下几句指令,即可创建记录对象与数据的添加
from tensorboardX import SummaryWriter
# 创建writer对象
writer = SummaryWriter('logs/tmp')
# 添加曲线，并且可以使用'/'进行多级标题的指定
writer.add_scalar('loss/total_loss', loss.data[0], total_iter)
writer.add_scalar('loss/rpn_loss', rpn_loss.data[0], total_iter)
'''
添加TensorBoardX指令后,则将在logs/tmp文件夹下生成events开头的记录文件,然后使用TensorBoard在终端中开启Web服务。
    tensorboard  --logdir=logs/tmp
    TensorBoard 1.9.0 at http://pytorch:6006 (Press CTRL+C to quit)
在浏览器端输入上述指令下方的网址http://pytorch:6006,即可看到数据的可视化效果。
'''

'''
2.Visdom由Facebook团队开发,是一个非常灵活的可视化工具,可用于多种数据的创建、组织和共享,支持NumPy、Torch与PyTorch数据,目的是促进远
程数据的可视化,支持科学实验。
Visdom可以通过pip指令完成安装:
    pip install visdom
使用如下指令可以开启visdom服务,该服务基于Web,并默认使用8097端口。
    python -m visdom.server
'''
# 下面实现一个文本、曲线及图片的可视化示例
import torch
import visdom
# 创建visdom客户端，使用默认端口8097，环境为first，环境的作用是对可视化的空间进行分区
vis = visdom.Visdom(env='first')
# vis对象有text(),line(),和image()等函数，其中的win参数代表了显示的窗格(pane)的名字
vis.text('first visdom', win='text1')
# 在此使用append为真来进行增加text，否则会覆盖之前的text
vis.text('hello pytorch', win='text1',append=True)
# 绘制y=-i^2+20*i+1的曲线，opts可以进行标题、坐标轴标签等的配置
for i in rangge(20):
    vis.line(X=torch.FloatTensor([i]), Y=torch.FloatTensor([-i**2+20*i+1]),
             opts={'title': 'y=-x^2+20x+1'}, win='loss', update='append')
# 可视化一张随机图片
vis.image(torch.randn(3,256,256), win='random_image')