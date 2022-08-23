# -*- conding:utf-8 -*-
#  @FileName    :DrawHistogram.py
#  @Time        :2022/3/21_19:47
#  @Author      :ZhangJian
#  @Description :
# -*-coding:utf8-*-#
import cv2
from matplotlib import pyplot as plt
from matplotlib import rcParams
config = {
    "font.family":'Arial',
}
rcParams.update(config)

img = cv2.imread(r'C:\Users\Administrator\Desktop\HHistorgam\55clahe.png',0)
gt = cv2.imread(r'C:\Users\Administrator\Desktop\HHistorgam\GT55.png',0)
mask1 = cv2.imread(r'C:\Users\Administrator\Desktop\HHistorgam\Unet55_pred.png',0)
mask2 = cv2.imread(r'C:\Users\Administrator\Desktop\HHistorgam\ResSE55_pred.png',0)
mask3 = cv2.imread(r'C:\Users\Administrator\Desktop\HHistorgam\PPM55_pred.png',0)
mask4 = cv2.imread(r'C:\Users\Administrator\Desktop\HHistorgam\DARU55_pred.png',0)

# 2x2展示
# plt.subplot(221), plt.imshow(img, 'gray')
# plt.subplot(222), plt.imshow(cv2.bitwise_and(img,img,mask = gt), 'gray')
# plt.subplot(223), plt.imshow(cv2.bitwise_and(img,img,mask = mask1), 'gray')
# plt.subplot(224), plt.imshow(cv2.bitwise_and(img,img,mask = mask2), 'gray')
# plt.show()

# cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate ]]) ->hist
# images：输入的图像
# channels：选择图像的通道
# mask：为None时处理整幅图像
# histSize：使用多少个bin，一般为256
# ranges：像素值范围[0,255]

# ========================= 第1图 =========================== #
hist_gt = cv2.calcHist([img],[0],gt,[256],[0,256])
hist_mask1 = cv2.calcHist([img],[0],mask1,[256],[0,256])
hist_mask2 = cv2.calcHist([img],[0],mask2,[256],[0,256])
hist_mask3 = cv2.calcHist([img],[0],mask3,[256],[0,256])
hist_mask4 = cv2.calcHist([img],[0],mask4,[256],[0,256])

height1 = [0]*256
height2 = [0]*256
height3 = [0]*256
height4 = [0]*256

for i in range(256):
    height1[i] = hist_mask1[i][0]
    height2[i] = hist_mask2[i][0]
    height3[i] = hist_mask3[i][0]
    height4[i] = hist_mask4[i][0]
# ========================= 第1图 =========================== #

# ========================= 第2图 =========================== #
img2 = cv2.imread(r'C:\Users\Administrator\Desktop\test_clahe\49_141.png',0)
mask21 = cv2.imread(r'C:\Users\Administrator\Desktop\HHistorgam\141_pred_unet11.png',0)
mask22 = cv2.imread(r'C:\Users\Administrator\Desktop\HHistorgam\ResSE141_pred.png',0)
mask23 = cv2.imread(r'C:\Users\Administrator\Desktop\HHistorgam\PPM141_pred.png',0)
mask24 = cv2.imread(r'C:\Users\Administrator\Desktop\HHistorgam\DARU141_pred.png',0)

hist_mask21 = cv2.calcHist([img2],[0],mask21,[256],[0,256])
hist_mask22 = cv2.calcHist([img2],[0],mask22,[256],[0,256])
hist_mask23 = cv2.calcHist([img2],[0],mask23,[256],[0,256])
hist_mask24 = cv2.calcHist([img2],[0],mask24,[256],[0,256])

height21 = [0]*256
height22 = [0]*256
height23 = [0]*256
height24 = [0]*256

for i in range(256):
    height21[i] = hist_mask21[i][0]
    height22[i] = hist_mask22[i][0]
    height23[i] = hist_mask23[i][0]
    height24[i] = hist_mask24[i][0]
# ========================= 第2图 =========================== #

# ========================= 第3图 =========================== #
img3 = cv2.imread(r'C:\Users\Administrator\Desktop\HHistorgam\212clahe.png',0)
mask31 = cv2.imread(r'C:\Users\Administrator\Desktop\HHistorgam\Unet212_pred.png',0)
mask32 = cv2.imread(r'C:\Users\Administrator\Desktop\HHistorgam\ResSE212_pred.png',0)
mask33 = cv2.imread(r'C:\Users\Administrator\Desktop\HHistorgam\PPM212_pred.png',0)
mask34 = cv2.imread(r'C:\Users\Administrator\Desktop\HHistorgam\DARU212_pred.png',0)

hist_mask31 = cv2.calcHist([img3],[0],mask31,[256],[0,256])
hist_mask32 = cv2.calcHist([img3],[0],mask32,[256],[0,256])
hist_mask33 = cv2.calcHist([img3],[0],mask33,[256],[0,256])
hist_mask34 = cv2.calcHist([img3],[0],mask34,[256],[0,256])

height31 = [0]*256
height32 = [0]*256
height33 = [0]*256
height34 = [0]*256

for i in range(256):
    height31[i] = hist_mask31[i][0]
    height32[i] = hist_mask32[i][0]
    height33[i] = hist_mask33[i][0]
    height34[i] = hist_mask34[i][0]
# ========================= 第3图 =========================== #

# ========================= 第4图 =========================== #
img4 = cv2.imread(r'C:\Users\Administrator\Desktop\HHistorgam\302_le.png',0)
mask41 = cv2.imread(r'C:\Users\Administrator\Desktop\HHistorgam\Unet302_pred.png',0)
mask42 = cv2.imread(r'C:\Users\Administrator\Desktop\HHistorgam\ResSE302_pred.png',0)
mask43 = cv2.imread(r'C:\Users\Administrator\Desktop\HHistorgam\PPM302_pred.png',0)
mask44 = cv2.imread(r'C:\Users\Administrator\Desktop\HHistorgam\DARU302_pred.png',0)

hist_mask41 = cv2.calcHist([img4],[0],mask41,[256],[0,256])
hist_mask42 = cv2.calcHist([img4],[0],mask42,[256],[0,256])
hist_mask43 = cv2.calcHist([img4],[0],mask43,[256],[0,256])
hist_mask44 = cv2.calcHist([img4],[0],mask44,[256],[0,256])

height41 = [0]*256
height42 = [0]*256
height43 = [0]*256
height44 = [0]*256

for i in range(256):
    height41[i] = hist_mask41[i][0]
    height42[i] = hist_mask42[i][0]
    height43[i] = hist_mask43[i][0]
    height44[i] = hist_mask44[i][0]
print(height41)
# ========================= 第4图 =========================== #

# ========================= 作 图 =========================== #
font1 = {'size':15}
font2 = {'size':13}
# 图1:

plt.subplot(211)
plt.bar(x=[i for i in range(256)],height=height1,color='#2A557F',label='UNet')
plt.bar(x=[i for i in range(256)],height=height2,color='#45BC9C',label='UNet+Res-SE')
plt.bar(x=[i for i in range(256)],height=height3,color='#FFCD6E',label='UNet+PPM')
plt.bar(x=[i for i in range(256)],height=height4,color='#ea6785',label='DARUNet')
plt.title("Case9 Slice12",font1)
plt.xlabel("Gray Level",font2)
plt.ylabel("Number of Pixels",font2)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_color('gray')
ax.spines['top'].set_color('gray')
plt.xlim([0,175])
plt.legend()

# plt.axvline(x=25,lw=1.5,color='r',linestyle='--')
# plt.axvline(x=120,lw=1.5,color='r',linestyle='--')

# 图2

plt.subplot(212)
plt.bar(x=[i for i in range(256)],height=height22,color='#45BC9C',label='UNet+Res-SE')
plt.bar(x=[i for i in range(256)],height=height23,color='#FFCD6E',label='UNet+PPM')
plt.bar(x=[i for i in range(256)],height=height24,color='#ea6785',label='DARUNet')
plt.bar(x=[i for i in range(256)],height=height21,color='#2A557F',label='UNet')
plt.title("Case25 Slice10",font1)
plt.xlabel("Gray Level",font2)
plt.ylabel("Number of Pixels",font2)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_color('gray')
ax.spines['top'].set_color('gray')
plt.xlim([0,175])
plt.legend()

# plt.axvline(x=25,lw=1.5,color='r',linestyle='--')
# plt.axvline(x=120,lw=1.5,color='r',linestyle='--')

# 图3

# plt.subplot(211)
# plt.bar(x=[i for i in range(256)],height=height31,color='#2A557F',label='UNet')
# plt.bar(x=[i for i in range(256)],height=height32,color='#45BC9C',label='UNet+Res-SE')
# plt.bar(x=[i for i in range(256)],height=height33,color='#FFCD6E',label='UNet+PPM')
# plt.bar(x=[i for i in range(256)],height=height34,color='#ea6785',label='DARUNet')
# plt.title("Case50 Slice15",font1)
# plt.xlabel("Gray Level",font2)
# plt.ylabel("Number of Pixels",font2)
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(1.5)
# ax.spines['left'].set_linewidth(1.5)
# ax.spines['right'].set_color('gray')
# ax.spines['top'].set_color('gray')
# plt.xlim([0,175])
# plt.legend()

# plt.axvline(x=25,lw=1.5,color='r',linestyle='--')
# plt.axvline(x=120,lw=1.5,color='r',linestyle='--')

# 图4

# plt.subplot(212)
# plt.bar(x=[i for i in range(256)],height=height44,color='#ea6785',label='DARUNet')
# plt.bar(x=[i for i in range(256)],height=height42,color='#45BC9C',label='UNet+Res-SE')
# plt.bar(x=[i for i in range(256)],height=height41,color='#2A557F',label='UNet')
# plt.bar(x=[i for i in range(256)],height=height43,color='#FFCD6E',label='UNet+PPM')
# plt.title("Case73 Slice17",font1)
# plt.xlabel("Gray Level",font2)
# plt.ylabel("Number of Pixels",font2)
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(1.5)
# ax.spines['left'].set_linewidth(1.5)
# ax.spines['right'].set_color('gray')
# ax.spines['top'].set_color('gray')
# plt.xlim([50,225])
# plt.legend()

# plt.axvline(x=25,lw=1.5,color='r',linestyle='--')
# plt.axvline(x=120,lw=1.5,color='r',linestyle='--')

# plt.plot(hist_mask1,color='#969696',label='UNet')
# plt.plot(hist_mask2,color='#007CE4',label='UNet+Res-SE')
# plt.plot(hist_mask3,color='#35547f',label='UNet+PPM')
# plt.plot(hist_mask4,color='#dd2333',label='DARUNet')
# plt.plot(hist_gt,color='#E65971',label='GroundTruth')
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['figure.dpi'] = 200
plt.show()
