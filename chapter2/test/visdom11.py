import torch
import visdom
# 创建visdom客户端，使用默认端口8097，环境为first，环境的作用是对可视化的空间进行分区
vis = visdom.Visdom(env='first')
# vis对象有text(),line(),和image()等函数，其中的win参数代表了显示的窗格(pane)的名字
vis.text('first visdom', win='text1')
# 在此使用append为真来进行增加text，否则会覆盖之前的text
vis.text('hello pytorch', win='text1',append=True)
# 绘制y=-i^2+20*i+1的曲线，opts可以进行标题、坐标轴标签等的配置
for i in range(20):
    vis.line(X=torch.FloatTensor([i]), Y=torch.FloatTensor([-i**2+20*i+1]),
             opts={'title': 'y=-x^2+20x+1'}, win='loss', update='append')
# 可视化一张随机图片
vis.image(torch.randn(3,256,256), win='random_image')