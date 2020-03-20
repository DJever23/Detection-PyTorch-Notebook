import os
import sys
from collections import Counter
import time

import matplotlib.pyplot as plt
import numpy as np

from IPython.display import display

class Evaluator:
    def GetPascalVOCMetrics(self,
                            cfg,
                            classes, 
                            gt_boxes,
                            num_pos,
                            det_boxes):

        ret = []
        groundTruths = []
        detections = []
        
        for c in classes:
            dects = det_boxes[c]#取当前类别的所有检测框
            print('dects[c]',dects)
            #[[12.0, 58.0, 53.0, 96.0, 0.87, '1'], [51.0, 88.0, 152.0, 191.0, 0.98, '1'], [243.0, 546.0, 298.0, 583.0, 0.83, '1']]
            gt_class = gt_boxes[c]#取当前类别的所有GT框
            #{'1': [[14.0, 56.0, 50.0, 100.0, 0], [50.0, 90.0, 150.0, 189.0, 0], [458.0, 657.0, 580.0, 742.0, 0]]}
            npos = num_pos[c]#取当前类别GT框的数量
            dects = sorted(dects, key=lambda conf: conf[5], reverse=True)
            #按[12.0, 58.0, 53.0, 96.0, 0.87, '1']的第6个值进行排序，且是从大到小排序
            print('dects',dects)
            TP = np.zeros(len(dects))
            FP = np.zeros(len(dects))
                
            for d in range(len(dects)):

                iouMax = sys.float_info.min#用sys.float_info来查看浮点数的信息,2.2250738585072014e-308
                #print('iouMaxs',iouMax)
                if dects[d][-1] in gt_class:
                    for j in range(len(gt_class[dects[d][-1]])):#GT框的数量
                        iou = Evaluator.iou(dects[d][:4], gt_class[dects[d][-1]][j][:4])
                        #当前类别的当前检测框与当前类别每个GT框的iou
                        if iou > iouMax:
                            iouMax = iou
                            jmax = j#当前检测狂与所有的GT框的iou最大时框的索引

                    if iouMax >= cfg['iouThreshold']:
                        #当前检测框与素有GT框取得的最大IOU>=iou阈值，这里是0.5，也即检测框匹配到了某个GT框
                        if gt_class[dects[d][-1]][jmax][4] == 0:
                            TP[d] = 1
                            gt_class[dects[d][-1]][jmax][4] = 1
                            #匹配成功后将不能再被其他框匹配，这里可以先通过将iou阈值调高来过滤一些iou较低的检测框

                        else:
                            FP[d] = 1
                    else:
                        FP[d] = 1
                else:
                    FP[d] = 1
            print('FP',FP)#[0,0,1]
            print('TP',TP)#[1,1,0]
            acc_FP = np.cumsum(FP)#[0,0,1]，这里为什么要累加？
            acc_TP = np.cumsum(TP)#[1,2,2]
            print('acc_FP',acc_FP)
            print('acc_TP',acc_TP)
            rec = acc_TP / npos#[0.33333333 0.66666667 0.66666667]
            print('rec',rec)
            prec = np.divide(acc_TP, (acc_FP + acc_TP))#acc_TP除以(acc_FP + acc_TP)，即[1,2,2]除以[1,2,3]
            print('prec',prec)#[1.         1.         0.66666667]
            print(' ')
            [ap, mpre, mrec, ii] = Evaluator.CalculateAveragePrecision(rec, prec)
            print('ap,mpre,mrec,ii',[ap, mpre, mrec, ii])
            r = {
                'class': c,
                'precision': prec,
                'recall': rec,
                'AP': ap,
                'interpolated precision': mpre,
                'interpolated recall': mrec,
                'total positives': npos,
                'total TP': np.sum(TP),
                'total FP': np.sum(FP),
            }
            ret.append(r)
        print('ret,classes',ret,classes)
        '''
        [{'class': 'class1', 'precision': array([1.        , 1.        , 0.66666667]), 
        'recall': array([0.33333333, 0.66666667, 0.66666667]), 'AP': 0.6666666666666666, 
        'interpolated precision': [1.0, 1.0, 1.0, 0.6666666666666666], 
        'interpolated recall': [0, 0.3333333333333333, 0.6666666666666666, 0.6666666666666666], 
        'total positives': 3, 'total TP': 2.0, 'total FP': 1.0}, 
        {'class': 'class2', 'precision': array([1.        , 1.        , 0.66666667]), 
        'recall': array([0.5, 1. , 1. ]), 'AP': 1.0, 
        'interpolated precision': [1.0, 1.0, 1.0, 0.6666666666666666], 
        'interpolated recall': [0, 0.5, 1.0, 1.0], 
        'total positives': 2, 'total TP': 2.0, 'total FP': 1.0}] 
        
        ['class1', 'class2']
        '''
        return ret, classes

    @staticmethod
    def CalculateAveragePrecision(rec, prec):#这一段的逻辑还需好好研究
        mrec = []
        mrec.append(0)
        [mrec.append(e) for e in rec]
        mrec.append(1)
        #mrec=[0,0.33333333,0.66666667,0.66666667,1]
        print('mrec',mrec)
        mpre = []
        mpre.append(0)
        [mpre.append(e) for e in prec]
        mpre.append(0)
        #mpre=[0,1.,1.,0.66666667,0]
        print('mpre',mpre)

        for i in range(len(mpre) - 1, 0, -1):
            #range(4,0,-1)意思是从4开始，倒序取到0的(但是不包括0),也就是以次取[4,3,2,1]
            #print('range(len(mpre) - 1, 0, -1)',i)
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        print('sort mpre',mpre)#[1.0, 1.0, 1.0, 0.6666666666666666, 0]
        ii = []
        for i in range(len(mrec) - 1):
            if mrec[i+1] != mrec[i]:
                ii.append(i + 1)
        print('ii',ii)#[1,2,4]
        ap = 0
        for i in ii:
            print('(mrec[i] - mrec[i - 1]) * mpre[i]',(mrec[i] - mrec[i - 1]) * mpre[i])
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
            print('ap',ap)
        return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]

    @staticmethod
    def iou(boxA, boxB):
        # if boxes dont intersect,boxA为检测狂,B为GT框
        if Evaluator._boxesIntersect(boxA, boxB) is False:#两个框有交集返回True，否则返回False
            return 0
        interArea = Evaluator._getIntersectionArea(boxA, boxB)#计算交集的面积
        union = Evaluator._getUnionAreas(boxA, boxB, interArea=interArea)#计算并集面积
        # intersection over union
        iou = interArea / union
        if iou < 0:
            import pdb
            pdb.set_trace()#断点调试
        assert iou >= 0
        return iou

    # boxA = (Ax1,Ay1,Ax2,Ay2)
    # boxB = (Bx1,By1,Bx2,By2)
    @staticmethod
    def _boxesIntersect(boxA, boxB):
        if boxA[0] > boxB[2]:
            return False  # boxA is right of boxB
        if boxB[0] > boxA[2]:
            return False  # boxA is left of boxB
        if boxA[3] < boxB[1]:
            return False  # boxA is above boxB
        if boxA[1] > boxB[3]:
            return False  # boxA is below boxB
        return True

    @staticmethod
    def _getIntersectionArea(boxA, boxB):

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # intersection area
        return (xB - xA + 1) * (yB - yA + 1)

    @staticmethod
    def _getUnionAreas(boxA, boxB, interArea=None):
        area_A = Evaluator._getArea(boxA)
        area_B = Evaluator._getArea(boxB)
        if interArea is None:
            interArea = Evaluator._getIntersectionArea(boxA, boxB)
        return float(area_A + area_B - interArea)

    @staticmethod
    def _getArea(box):
        return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
