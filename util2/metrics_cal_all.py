# -*- conding:utf-8 -*-
#  @FileName    :Metrics_EachCase_Avg.py
#  @Time        :2021/3/24_14:44
#  @Author      :ZhangJian
#  @Description : 测试集全体(All cases)作为一个对象来计算Dice

import cv2
import os
epsilon = 1e-7

pred_file = r'C:\Users\Administrator\Desktop\MultiResUnet'
mask_file = r'C:\Users\Administrator\Desktop\selected_image\test\mask'
f1 = os.listdir(pred_file)
f2 = os.listdir(mask_file)
f1.sort(key=lambda x: int(x[:-9]))
f2.sort(key=lambda x: int(x[:-4]))

iou, iou2 = [0, 0]
tp, tn, fp, fn = [0, 0, 0, 0]
for p1,p2 in zip(f1,f2):
    print(p1,p2)
    preds = os.path.join(pred_file,p1)
    masks = os.path.join(mask_file,p2)

    pred = cv2.imread(preds,0)
    mask = cv2.imread(masks,0)

    rows,cols = pred.shape[:2]
    # row-行 col-列
    # 灰度=0 ，黑色，阴性
    # 灰度!=0 ，白色，阳性

    for row in range(rows):
        for col in range(cols):
            a = pred[row,col]
            b = mask[row,col]
            # 预测正确
            if(a==b):
                # b为1，是阳性，TP
                if(b!=0):
                    tp += 1
                    iou += 1
                    iou2 += 1
                # TN
                else:
                    tn += 1
            #预测错误
            else:
                # b为1，是阳性，预测成阴性，FN
                if(b!=0):
                    fn += 1
                    iou2 += 1
                # b是0，是阴性，预测成阳性，FP
                else:
                    fp += 1
                    iou2 += 1

accuracy = (tp + tn) / (tp + tn + fp + fn)
Jaccard_Index = tp / (tp + fp + fn)
Iou = iou / iou2
Dice = 2 * tp / (2 * tp + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn) # 阳性目标敏感度
sensitivity = tp / (tp + fn + epsilon)
F1 = 2*(precision*recall) / (precision+recall)

print('ACC: %.4f' %accuracy)
print('sensitivity: %.4f' %sensitivity)
print('IoU: %.4f' %Iou)
print('JI: %.4f' %Jaccard_Index)
print('Recall: %.4f' %recall)
print('Precision: %.4f' %precision)
print('Dice: %.4f' % Dice)
print('F1: %.4f' % F1)
