'''
PyTorch将数据集的处理过程标准化,提供了Dataset基本的数据类,并在torchvision中提供了众多数据变换函数,数据加载的具体过程主要分为3
步:继承Dataset类，增加数据变换，继承Dataloader
'''
# 1.继承Dataset类
'''
对于数据集的处理,PyTorch提供了torch.utils.data.Dataset这个抽象类,在使用时只需要继承该类,并重写__len__()和__getitem()__函数,
即可以方便地进行数据集的迭代。
'''
from torch.utils.data import Dataset


class my_data(Dataset):
    def __init__(self,image_path, annotation_path, transform=None):
        # 初始化，读取数据集
    def __len__(self):
        # 获取数据集的总大小
    def __getitem__(self, id):
        # 对于指定的id，读取该数据并返回

# 对上述类进行实例化，即可进行迭代
dataset = my_data("your image path", "your annotation path")  # 实例化该类
for data in dataset:
    print(data)

# 2.数据的变换与增强：torchvision.transforms
'''
PyTorch提供了torchvision.transforms工具包,可以方便地进行图像缩放、裁剪、随机翻转、填充及张量的归一化等操作,操作对象是PIL的
Image或者Tensor。如果需要进行多个变换功能,可以利用transforms.Compose将多个变换整合起来,并且在实际使用时,通常会将变换操作集
成到Dataset的继承类中。
'''
from torchvision import transforms
# 将transforms集成到Dataset类中，使用Compose将多个变换整合到一起
data = my_data("your image path", "your annotation path", transforms=
               transforms.Compose([
                   transforms.Resize(256)  # 将图像最短边缩小至256，宽高比例不变
                   # 以0.5的概率随机翻转指定的PIL图像
                   transforms.RandomHorizontalFlip()
                   # 将PIL图像转为Tensor，元素区间从[0,255]归一到[0,1]
                   transforms.ToTensor()
                   # 进行mean与std为0.5的标准化
                   transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
               ]))

# 3.继承dataloader
'''
经过前两步已经可以获取每一个变换后的样本,但是仍然无法进行批量处理、随机选取等操作,因此还需要torch.utils.data.Dataloader类进一
步进行封装,使用方法如下例所示,该类需要4个参数,第1个参数是之前继承了Dataset的实例,第2个参数是批量batch的大小,第3个参数是是否打乱
数据参数,第4个参数是使用几个线程来加载数据。
'''
from torch.utils.data import DataLoader
# 使用Dataloader进一步封装Dataset
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

#dataloader是一个可迭代对象，对该实例进行迭代即可用于训练过程
iters_per_epoch = 500
data_iter = iter(dataloader)
for step in range(iters_per_epoch):
    data = next(data_iter)
    #将data用于训练网络即可
