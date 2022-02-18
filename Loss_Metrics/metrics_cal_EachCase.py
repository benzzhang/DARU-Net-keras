# -*- conding:utf-8 -*-
#  @FileName    :Metrics_EachCase_Avg.py
#  @Time        :2021/3/24_14:44
#  @Author      :ZhangJian
#  @Description : Calculate recall, precision and dice separately for each case, and finally calculate  the mean value

import cv2
import os
import numpy as np
epsilon = 1e-7

pred_file = r'C:\Users\Administrator\Desktop\empty'
mask_file = r'C:\Users\Administrator\Desktop\selected_image\each_mask'
f1 = os.listdir(pred_file)
f2 = os.listdir(mask_file)
f1.sort(key=lambda x: int(x))
f2.sort(key=lambda x: int(x))

Acc = []
JI =[]
Precision = []
Recall = []
Specificity = []
Dice = []
F1 = []

for p1,p2 in zip(f1,f2):
    print(p1,p2)
    preds = os.listdir(os.path.join(pred_file,p1))
    masks = os.listdir(os.path.join(mask_file,p2))

    preds.sort(key=lambda x: int(x[:-9]))
    masks.sort(key=lambda x: int(x[:-4]))
    tp, tn, fp, fn = [0, 0, 0, 0]
    for p,m in zip(preds,masks):
        pred_path = os.path.join(os.path.join(pred_file,p1), p)
        mask_path = os.path.join(os.path.join(mask_file,p2), m)

        pred = cv2.imread(pred_path,0)
        mask = cv2.imread(mask_path,0)

        rows,cols = pred.shape[:2]
        # row-行 col-列
        # gray level = 0 ，black，negative
        # gray level != 0 ，white，positive

        for row in range(rows):
            for col in range(cols):
                a = pred[row,col]
                b = mask[row,col]
                # Correct segmentation
                if(a==b):
                    # b - 1，positive，TP
                    if(b!=0):
                        tp += 1
                    # TN
                    else:
                        tn += 1
                # Wrong segmentation
                else:
                    # b - 1，positive，but seg:negative，FN
                    if(b!=0):
                        fn += 1
                    # b - 0，negative，but seg:positive，FP
                    else:
                        fp += 1
        #print(p+' match '+m)

    print('tp的指标如下：%d' %tp)
    print('tn的指标如下：%d' %tn)
    print('fp的指标如下：%d' %fp)
    print('fn的指标如下：%d' %fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn+epsilon)
    Jaccard_Index = tp / (tp + fp + fn+epsilon)
    dice = 2 * tp / (2 * tp + fp + fn+epsilon)
    precision = tp / (tp + fp+epsilon)
    recall = tp / (tp + fn+epsilon)
    f1 = 2 * (precision * recall) / (precision + recall+epsilon)

    print('case%s的指标如下：' %p1)
    print('JI: %.4f' %Jaccard_Index)
    print('Recall: %.4f' %recall)
    print('Precision: %.4f' %precision)
    print('Dice: %.4f' % dice)
    print(' ')
    Acc.append(accuracy)
    JI.append(Jaccard_Index)
    Precision.append(precision)
    Recall.append(recall)
    Dice.append(dice)
    F1.append(f1)

Acc_mean = np.mean(Acc)
JI_mean = np.mean(JI)
Precision_mean = np.mean(Precision)
Recall_mean = np.mean(Recall)
Dice_mean = np.mean(Dice)
Dice_std = np.std(Dice, ddof=1)
F11 = 2*(Precision_mean*Recall_mean) / (Precision_mean+Recall_mean)
F1_mean = np.mean(F1)
print('JI_mean: %.4f' % JI_mean)
print('Acc_mean: %.4f' % Acc_mean)
print('Dice_mean: %.4f' % Dice_mean)
print('Dice_std: %.4f' % Dice_std)
print('Recall_mean: %.4f' % Recall_mean)
print('Precision_mean: %.4f' % Precision_mean)
print('F1_mean: %.4f' % F1_mean)

recall_np = np.array(Recall)
precision_np = np.array(Precision)
recall_np_4f = np.round(recall_np,4)
precision_np_4f = np.round(precision_np,4)
recall_list = list(recall_np_4f)
precision_list = list(precision_np_4f)
print(Dice)