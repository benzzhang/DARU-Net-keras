# -*- conding:utf-8 -*-
#  @FileName    :DrawContours.py
#  @Time        :2020/12/22_11:01
#  @Author      :ZhangJian
#  @Description : Draw the contours of the prediction results on images

import cv2
import os

save_path = r'C:\Users\Administrator\Desktop\emerged_pics\GT_PPM3'

PREDS = r'C:\Users\Administrator\Desktop\U-Net\Unet12'  #red
PREDS2 = r'C:\Users\Administrator\Desktop\U-Net_PPM\Unet_PPM7' # yellow

MASK = r'C:\Users\Administrator\Desktop\selected_image\test\mask'  #green
IMAGE = r'C:\Users\Administrator\Desktop\test_image'

k1 = os.listdir(PREDS)
k1.sort(key=lambda x: int(x[:-9]))

k2 = os.listdir(MASK)
k2.sort(key=lambda x: int(x[:-4]))

k3 = os.listdir(IMAGE)
k3.sort(key=lambda x: int(x[:-4]))

k4 = os.listdir(PREDS2)
k4.sort(key=lambda x: int(x[:-9]))

for m1,m2,m3,m4 in zip(k1,k2,k3,k4):

    s1 = os.path.join(PREDS, m1)
    s2 = os.path.join(MASK, m2)
    s3 = os.path.join(IMAGE, m3)
    s4 = os.path.join(PREDS2, m4)

    print(m1+' match '+m2+' match '+m3)

    image = cv2.imread(s3)
    preds = cv2.imread(s1, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(s2, cv2.IMREAD_GRAYSCALE)
    preds2 = cv2.imread(s4,cv2.IMREAD_GRAYSCALE)

    contours, hierarchy = cv2.findContours(preds, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours2, hierarchy2 = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours3, hierarchy3 = cv2.findContours(preds2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # GT drawn last
    #cv2.drawContours(image, contours, -1, (0, 0, 255), 1)   #red
    cv2.drawContours(image, contours3, -1, (0, 255, 255), 1)  #yellow
    cv2.drawContours(image, contours2, -1, (0, 255, 0), 1)  #green, mask
    cv2.imwrite(os.path.join(save_path, m1), image)

k = cv2.waitKey(0)