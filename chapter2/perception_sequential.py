import torch
from torch import nn

'''
当模型中只是简单的前馈网络时,即上一层的输出直接作为下一层的输
入,这时可以采用nn.Sequential()模块来快速搭建模型,而不必手动在
forward()函数中一层一层地前向传播。因此,如果想快速搭建模型而不考
虑中间过程的话,推荐使用nn.Sequential()模块。
'''
class Perception(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(Perception, self).__init__()
        self.layer = nn.Sequential(
          nn.Linear(in_dim, hid_dim),
          nn.Sigmoid(),
          nn.Linear(hid_dim, out_dim),
          nn.Sigmoid()
)
    def forward(self, x):
        y = self.layer(x)
        return y

if __name__ == '__main__':
    model = Perception(100, 1000, 10).cuda()  # 构建类的实例，并表明在CUDA上
    print('model', model)
    input = torch.randn(100).cuda()
    output = model(input)  # 将输入传入实例化的模型
    print(output.shape)
    print(output)
