import os
import sys
import yaml
#yaml是一个专门用来写配置文件的语言
sys.path.insert(0, os.path.join(os.getcwd(), 'lib'))
#sys.path.insert(0,'/path')新添加的目录会优先于其他目录被import检查
#os.getcwd()方法用于返回当前工作目录,os.path.join()将前后路径组合
from detection import detections, plot_save_result

conf_path = './conf/conf.yaml'
with open(conf_path, 'r', encoding='utf-8') as f:
    data=f.read()
print(data)
print('type(data): ', type(data))# <class 'str'>
cfg = yaml.load(data)
print(cfg)
print('type(cfg): ', type(cfg))#<class 'dict'>
#python通过open方式读取文件数据，再通过load函数将数据转化为列表或字典；

gtFolder = 'data/groundtruths'
detFolder = 'data/detections'
savePath = 'data/results'

results, classes = detections(cfg, gtFolder, detFolder, savePath)
plot_save_result(cfg, results, classes, savePath)
