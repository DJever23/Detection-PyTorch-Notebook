import os
from Evaluator import *
import pdb

def getGTBoxes(cfg, GTFolder):

    files = os.listdir(GTFolder)
    files.sort()
    print(files)

    classes = []
    num_pos = {}
    gt_boxes = {}
    for f in files:
        nameOfImage = f.replace(".txt", "")
        #print('nameOfImage',nameOfImage)
        #str.replace(old, new[, max])返回字符串中的 old（旧字符串）
        #替换成 new(新字符串)后生成的新字符串，如果指定第三个参数max，则替换不超过max次。
        fh1 = open(os.path.join(GTFolder, f), "r")
        
        for line in fh1:
            line = line.replace("\n", "")
            #print('line:',line)#line: class1 14 56 50 100
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            #Python split() 通过指定分隔符对字符串进行切片，如果参数 num 有指定值，则分隔 num+1 个子字符串
            #print('splitLine:',splitLine)#['class1', '14', '56', '50', '100']

            cls = (splitLine[0])  # class
            left = float(splitLine[1])
            top = float(splitLine[2])
            right = float(splitLine[3])
            bottom = float(splitLine[4])
            #print(cls,left)
            one_box = [left, top, right, bottom, 0]
              
            if cls not in classes:
                classes.append(cls)
                gt_boxes[cls] = {}
                num_pos[cls] = 0

            num_pos[cls] += 1

            if nameOfImage not in gt_boxes[cls]:
                gt_boxes[cls][nameOfImage] = []
                #print('in if')
                #print('gt_boxes',gt_boxes)
            gt_boxes[cls][nameOfImage].append(one_box)
            #print('gt_boxes.append',gt_boxes)
        print(classes, gt_boxes, num_pos)
        fh1.close()
    return gt_boxes, classes, num_pos

def getDetBoxes(cfg, DetFolder):

    files = os.listdir(DetFolder)
    files.sort()

    det_boxes = {}
    for f in files:
        nameOfImage = f.replace(".txt", "")
        fh1 = open(os.path.join(DetFolder, f), "r")

        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")

            cls = (splitLine[0])  # class
            left = float(splitLine[1])
            top = float(splitLine[2])
            right = float(splitLine[3])
            bottom = float(splitLine[4])
            score = float(splitLine[5])
            one_box = [left, top, right, bottom, score, nameOfImage]

            if cls not in det_boxes:
                det_boxes[cls]=[]
            det_boxes[cls].append(one_box)
        print('det_boxes',det_boxes)
        fh1.close()
    return det_boxes

def detections(cfg,
               gtFolder,
               detFolder,
               savePath,
               show_process=True):
    

    gt_boxes, classes, num_pos = getGTBoxes(cfg, gtFolder)
    '''
    gt_boxes：{'class1': {'1': [[14.0, 56.0, 50.0, 100.0, 0], [50.0, 90.0, 150.0, 189.0, 0], [458.0, 657.0, 580.0, 742.0, 0]]}, 
                'class2': {'1': [[345.0, 894.0, 432.0, 940.0, 0], [590.0, 354.0, 675.0, 420.0, 0]]}}
    classes：['class1', 'class2']
    num_pos：{'class1': 3, 'class2': 2}
    '''

    det_boxes = getDetBoxes(cfg, detFolder)
    '''
    det_boxes {'class1': [[12.0, 58.0, 53.0, 96.0, 0.87, '1'], [51.0, 88.0, 152.0, 191.0, 0.98, '1'], [243.0, 546.0, 298.0, 583.0, 0.83, '1']], 
    'class2': [[345.0, 898.0, 431.0, 945.0, 0.67, '1'], [597.0, 346.0, 674.0, 415.0, 0.45, '1'], [99.0, 345.0, 150.0, 426.0, 0.96, '1']]}
    '''
    
    evaluator = Evaluator()#实例化Evaluator类

    return evaluator.GetPascalVOCMetrics(cfg, classes, gt_boxes, num_pos, det_boxes)

def plot_save_result(cfg, results, classes, savePath):
    
    
    plt.rcParams['savefig.dpi'] = 80##图片像素
    plt.rcParams['figure.dpi'] = 130#分辨率

    acc_AP = 0
    validClasses = 0
    fig_index = 0

    for cls_index, result in enumerate(results):
        if result is None:
            raise IOError('Error: Class %d could not be found.' % (cls_index+1))

        cls = result['class']
        precision = result['precision']
        recall = result['recall']
        average_precision = result['AP']
        acc_AP = acc_AP + average_precision
        mpre = result['interpolated precision']
        mrec = result['interpolated recall']
        npos = result['total positives']
        total_tp = result['total TP']
        total_fp = result['total FP']

        fig_index+=1
        plt.figure(fig_index)
        plt.plot(recall, precision, cfg['colors'][cls_index], label='Precision')#“label”指定线条的标签
        plt.xlabel('recall')
        plt.ylabel('precision')
        ap_str = "{0:.2f}%".format(average_precision * 100)
        plt.title('Precision x Recall curve \nClass: %s, AP: %s' % (str(cls), ap_str))
        plt.legend(loc='lower left',shadow=True)#用于给图像加图例，图例是集中于地图一角或一侧的地图上各种符号和颜色所代表内容与指标的说明，有助于更好的认识地图。
        plt.grid()#绘制网格
        plt.savefig(os.path.join(savePath, cls + '.png'))
        plt.show()
        plt.pause(0.05)#显示0.05秒后自动关闭


    mAP = acc_AP / fig_index
    mAP_str = "{0:.2f}%".format(mAP * 100)
    print('mAP: %s' % mAP_str)
    
