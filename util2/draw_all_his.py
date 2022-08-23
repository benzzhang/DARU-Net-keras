# -*- conding:utf-8 -*-
#  @FileName    :DrawHistogram.py
#  @Time        :2022/3/21_19:47
#  @Author      :ZhangJian
#  @Description :
# -*-coding:utf8-*-#
import cv2
from matplotlib import pyplot as plt
import os

imgpath = r'C:\Users\Administrator\Desktop\histogram\image'
filepath1 = r'C:\Users\Administrator\Desktop\histogram\mask\GT'
filepath2 = r'C:\Users\Administrator\Desktop\histogram\mask\SE4'
filepath3 = r'C:\Users\Administrator\Desktop\histogram\mask\Unet12'

filenames = os.listdir(imgpath)
filenames.sort(key=lambda x: int(x[:-4]))
filenames2 = os.listdir(filepath2)
filenames2.sort(key=lambda x: int(x[:-9]))

x = cv2.imread(r'C:\Users\Administrator\Desktop\histogram\mask\GT\0.png', 0)
y = cv2.imread(r'C:\Users\Administrator\Desktop\histogram\mask\GT\0.png', 0)
hist_mask_GT = cv2.calcHist([x], [0], y, [256], [0, 256])
hist_mask_PRED = cv2.calcHist([x], [0], y, [256], [0, 256])
hist_mask_PRED2 = cv2.calcHist([x], [0], y, [256], [0, 256])

for f in filenames:
    # 找img 和 GT_mask
    img_path = os.path.join(imgpath, f)
    mask_path = os.path.join(filepath1,f)
    print(img_path)
    print(mask_path)
    img = cv2.imread(img_path, 0)
    mask = cv2.imread(mask_path, 0)
    hist_mask_GT += cv2.calcHist([img], [0], mask, [256], [0, 256])
print(hist_mask_GT)

for f,fx in zip(filenames,filenames2):
    # 找img 和 mask
    img_path = os.path.join(imgpath, f)
    mask_path = os.path.join(filepath2,fx)
    img = cv2.imread(img_path, 0)
    mask = cv2.imread(mask_path, 0)
    hist_mask_PRED += cv2.calcHist([img], [0], mask, [256], [0, 256])
print(hist_mask_PRED)

for f,fx in zip(filenames,filenames2):
    # 找img 和 mask
    img_path = os.path.join(imgpath, f)
    mask_path = os.path.join(filepath3,fx)
    img = cv2.imread(img_path, 0)
    mask = cv2.imread(mask_path, 0)
    hist_mask_PRED2 += cv2.calcHist([img], [0], mask, [256], [0, 256])
print(hist_mask_PRED2)

plt.plot(hist_mask_GT,color='#FF7E76',label='GroundTruth') # 红
plt.plot(hist_mask_PRED,color='#007CE4',label='UNet+SE') # 蓝
plt.plot(hist_mask_PRED2,color='#1d953f',label='UNet') # 绿
plt.legend()
plt.xlim([0,256])
plt.show()