# -*- conding:utf-8 -*-
#  @FileName    :threshold.py
#  @Time        :2021/1/18_15:41
#  @Author      :ZhangJian
#  @Description :

import cv2
import os

# input folder
pred_file = r'C:\Users\Administrator\Desktop\preds_RES_SE10'
f = os.listdir(pred_file)

for fs in f:
    path1 = os.path.join(pred_file,fs)

    pred = cv2.imread(path1,0)

    ret, dst = cv2.threshold(pred, 200, 255, cv2.THRESH_BINARY)

    # output folder
    path_ = r'C:\Users\Administrator\Desktop\222'
    path2 = os.path.join(path_,fs)
    cv2.imwrite(path2, dst)
