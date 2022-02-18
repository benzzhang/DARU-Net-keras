# -*- conding:utf-8 -*-
#  @FileName    :boxplot_DscOnly.py
#  @Time        :2021/10/8_15:58
#  @Author      :ZhangJian
#  @Description : 根据.xls文件输出boxplot,只含dsc

import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
plt.rc('font',family='Times New Roman')

# 从.XLS读取数据
filepath = r'C:\Users\Administrator\Desktop\age_ca125_HE4_D.xlsx'
wb = openpyxl.load_workbook(filepath,data_only=True)
sheet1 = wb.worksheets[0]
ws = wb.active

# 第N行
row_u = sheet1[7]
row_ag = sheet1[13]
row_led = sheet1[19]
row_hr = sheet1[23]
row_mul = sheet1[30]
row_daru = sheet1[36]

# i1-i6：横坐标标记
i1 = i2 = i3 = i4 = i5 = i6 = -1
j1 = j2 = j3 = j4 = j5 = j6 = 0

dice_u = dict()
dice_ag = dict()
dice_led = dict()
dice_hr = dict()
dice_mul = dict()
dice_daru = dict()

# 插入dice数据
for cell in row_u[1:]:
    i1 += 1
    # U-Net
    if i1%6 == 0 :
        j1 += 1
        dice_u[j1] = cell.value

for cell in row_ag[1:]:
    i2 += 1
    # Attention_U
    if i2%6 == 0 :
        j2 += 1
        dice_ag[j2] = cell.value

for cell in row_led[1:]:
    i3 += 1
    # LED
    if i3%6 == 0 :
        j3 += 1
        dice_led[j3] = cell.value

for cell in row_hr[1:]:
    i4 += 1
    # HR
    if i4%6 == 0 :
        j4 += 1
        dice_hr[j4] = cell.value

for cell in row_mul[1:]:
    i5 += 1
    # MUL
    if i5%6 == 0 :
        j5 += 1
        dice_mul[j5] = cell.value

for cell in row_daru[1:]:
    i6 += 1
    # DARU_NET
    if i6%6 == 0 :
        j6 += 1
        dice_daru[j6] = cell.value

dt1 = pd.DataFrame({
    'U-Net': dice_u,
    'Attention U-Net': dice_ag,
    'LEDNet': dice_led,
    'HRNet': dice_hr,
    'MultiResNet': dice_mul,
    'DARU-Net': dice_daru,
})

# # 插入recall数据
# i1 = i2 = i3 = i4 = i5 = i6 = -3
# for cell in row_u[1:]:
#     i1 += 1
#     if i1%6 == 0 :
#         j1 += 1
#         dice_u[j1] = cell.value
#
# for cell in row_ag[1:]:
#     i2 += 1
#     if i2%6 == 0 :
#         j2 += 1
#         dice_ag[j2] = cell.value
#
# for cell in row_led[1:]:
#     i3 += 1
#     if i3%6 == 0 :
#         j3 += 1
#         dice_led[j3] = cell.value
#
# for cell in row_hr[1:]:
#     i4 += 1
#     if i4%6 == 0 :
#         j4 += 1
#         dice_hr[j4] = cell.value
#
# for cell in row_mul[1:]:
#     i5 += 1
#     if i5%6 == 0 :
#         j5 += 1
#         dice_mul[j5] = cell.value
#
# for cell in row_daru[1:]:
#     i6 += 1
#     if i6%6 == 0 :
#         j6 += 1
#         dice_daru[j6] = cell.value
#
# dt2 = pd.DataFrame({
#     'U-Net': dice_u,
#     'Attention U-Net': dice_ag,
#     'LEDNet': dice_led,
#     'HRNet': dice_hr,
#     'MultiResNet': dice_mul,
#     'DARU-Net': dice_daru,
# })
#
# # 做分类标记
# dt2['is_dice'] = 'yes'
# for i in range(31,61):
#     dt2.loc[i,'is_dice'] = 'no'

# dt1---dice only
# dt2---dice(1-30) + recall(31-60)
print(dt1)
print(dt1.describe())

plt.ylabel("Value")
#plt.grid()
# means: GREEN TRIANGLE
plt.boxplot(x=dt1.values,labels=dt1.columns,whis=1.5,showmeans=True, sym='b+')
plt.axhline(y=0.8,ls=":",c="gray") #添加水平直线
plt.show()
