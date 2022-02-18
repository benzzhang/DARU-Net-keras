# -*- conding:utf-8 -*-
#  @FileName    :boxplot_SingleMul.py
#  @Time        :2021/10/10_14:55
#  @Author      :ZhangJian
#  @Description : 读取单发/多发的xls文件数据, 写入.csv, 从csv绘图: 同时包含单发/多发

import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
import seaborn as sns
import csv
plt.rc('font',family='Times New Roman')

f = open('data_analysis2.csv','w',encoding='utf-8')
csv_writer = csv.writer(f)
csv_writer.writerow(["Value", "Methods", "metrics"])

filepath = 'EachCaseMetric_单发.xlsx'
wb = openpyxl.load_workbook(filepath,data_only=True)
sheet1 = wb.worksheets[0]
ws = wb.active

row_u = sheet1[7]
row_ag = sheet1[13]
row_led = sheet1[19]
row_hr = sheet1[23]
row_mul = sheet1[30]
row_daru = sheet1[36]


i1 = i2 = i3 = i4 = i5 = i6 = -1
j1 = j2 = j3 = j4 = j5 = j6 = 0


for cell in row_u[1:]:
    i1 += 1
    if i1%6 == 0 :
        j1 += 1
        csv_writer.writerow([cell.value, "U-Net", "single"])

for cell in row_ag[1:]:
    i2 += 1
    if i2%6 == 0 :
        j2 += 1
        csv_writer.writerow([cell.value, "Attention U-Net", "single"])

for cell in row_led[1:]:
    i3 += 1
    if i3%6 == 0 :
        j3 += 1
        csv_writer.writerow([cell.value, "LEDNet", "single"])

for cell in row_hr[1:]:
    i4 += 1
    if i4%6 == 0 :
        j4 += 1
        csv_writer.writerow([cell.value, "HRNet", "single"])

for cell in row_mul[1:]:
    i5 += 1
    if i5%6 == 0 :
        j5 += 1
        csv_writer.writerow([cell.value, "MultiResNet", "single"])

for cell in row_daru[1:]:
    i6 += 1
    if i6%6 == 0 :
        j6 += 1
        csv_writer.writerow([cell.value, "DARU-Net", "single"])

filepath = 'EachCaseMetric_多发.xlsx'
wb = openpyxl.load_workbook(filepath,data_only=True)
sheet1 = wb.worksheets[0]
ws = wb.active

row_u2 = sheet1[7]
row_ag2 = sheet1[13]
row_led2 = sheet1[19]
row_hr2 = sheet1[23]
row_mul2 = sheet1[30]
row_daru2 = sheet1[36]

i1 = i2 = i3 = i4 = i5 = i6 = -1
for cell in row_u2[1:]:
    i1 += 1
    if i1%6 == 0 :
        j1 += 1
        csv_writer.writerow([cell.value, "U-Net", "mul"])

for cell in row_ag2[1:]:
    i2 += 1
    if i2%6 == 0 :
        j2 += 1
        csv_writer.writerow([cell.value, "Attention U-Net", "mul"])

for cell in row_led2[1:]:
    i3 += 1
    if i3%6 == 0 :
        j3 += 1
        csv_writer.writerow([cell.value, "LEDNet", "mul"])

for cell in row_hr2[1:]:
    i4 += 1
    if i4%6 == 0 :
        j4 += 1
        csv_writer.writerow([cell.value, "HRNet", "mul"])

for cell in row_mul2[1:]:
    i5 += 1
    if i5%6 == 0 :
        j5 += 1
        csv_writer.writerow([cell.value, "MultiResNet", "mul"])

for cell in row_daru2[1:]:
    i6 += 1
    if i6%6 == 0 :
        j6 += 1
        csv_writer.writerow([cell.value, "DARU-Net", "mul"])

f.close()

dice_recall = pd.read_csv('data_analysis2.csv')

sns.set_style("whitegrid")
ax = sns.boxplot(x="Methods", y="Value", hue="metrics", data=dice_recall,
                 palette="Set3")

#ax = sns.swarmplot(x="Methods", y="Value", hue="metrics", data=dice_recall)
#plt.axhline(y=0.8,ls=":",c="black")#添加水平直线
#plt.grid(ls=":",c='b',)#打开坐标网格

#plt.title('Box plot of DSC and Recall for all test cases')
plt.xlabel('')
plt.show()