# -*- conding:utf-8 -*-
#  @FileName    :boxplot_DscRePre.py
#  @Time        :2021/10/9_14:29
#  @Author      :ZhangJian
#  @Description : 根据.xls文件输出boxplot, 包含dsc,recall,precision

import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
import seaborn as sns
import csv

# .CSV用来保存数据并绘boxplot图
f = open('../data_analysis.csv','w',encoding='utf-8')
csv_writer = csv.writer(f)
csv_writer.writerow(["Value", "Methods", "metrics"])

# 从.xls文件读取数据保存到.CSV
filepath = '../EachCaseMetric.xlsx'
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

# 从第1列开始，取DSC
for cell in row_u[1:]:
    i1 += 1
    if i1%6 == 0 :
        j1 += 1
        csv_writer.writerow([cell.value, "U-Net", "DSC"])

for cell in row_ag[1:]:
    i2 += 1
    if i2%6 == 0 :
        j2 += 1
        csv_writer.writerow([cell.value, "Attention U-Net", "DSC"])

for cell in row_led[1:]:
    i3 += 1
    if i3%6 == 0 :
        j3 += 1
        csv_writer.writerow([cell.value, "LEDNet", "DSC"])

for cell in row_hr[1:]:
    i4 += 1
    if i4%6 == 0 :
        j4 += 1
        csv_writer.writerow([cell.value, "HRNet", "DSC"])

for cell in row_mul[1:]:
    i5 += 1
    if i5%6 == 0 :
        j5 += 1
        csv_writer.writerow([cell.value, "MultiResNet", "DSC"])

for cell in row_daru[1:]:
    i6 += 1
    if i6%6 == 0 :
        j6 += 1
        csv_writer.writerow([cell.value, "DARU-Net", "DSC"])


# 取Recall
i1 = i2 = i3 = i4 = i5 = i6 = -3
for cell in row_u[1:]:
    i1 += 1
    if i1%6 == 0 :
        j1 += 1
        csv_writer.writerow([cell.value, "U-Net", "Recall"])

for cell in row_ag[1:]:
    i2 += 1
    if i2%6 == 0 :
        j2 += 1
        csv_writer.writerow([cell.value, "Attention U-Net", "Recall"])

for cell in row_led[1:]:
    i3 += 1
    if i3%6 == 0 :
        j3 += 1
        csv_writer.writerow([cell.value, "LEDNet", "Recall"])

for cell in row_hr[1:]:
    i4 += 1
    if i4%6 == 0 :
        j4 += 1
        csv_writer.writerow([cell.value, "HRNet", "Recall"])

for cell in row_mul[1:]:
    i5 += 1
    if i5%6 == 0 :
        j5 += 1
        csv_writer.writerow([cell.value, "MultiResNet", "Recall"])

for cell in row_daru[1:]:
    i6 += 1
    if i6%6 == 0 :
        j6 += 1
        csv_writer.writerow([cell.value, "DARU-Net", "Recall"])


# 取Precision
i1 = i2 = i3 = i4 = i5 = i6 = -2
for cell in row_u[1:]:
    i1 += 1
    if i1%6 == 0 :
        j1 += 1
        csv_writer.writerow([cell.value, "U-Net", "Precision"])

for cell in row_ag[1:]:
    i2 += 1
    if i2%6 == 0 :
        j2 += 1
        csv_writer.writerow([cell.value, "Attention U-Net", "Precision"])

for cell in row_led[1:]:
    i3 += 1
    if i3%6 == 0 :
        j3 += 1
        csv_writer.writerow([cell.value, "LEDNet", "Precision"])

for cell in row_hr[1:]:
    i4 += 1
    if i4%6 == 0 :
        j4 += 1
        csv_writer.writerow([cell.value, "HRNet", "Precision"])

for cell in row_mul[1:]:
    i5 += 1
    if i5%6 == 0 :
        j5 += 1
        csv_writer.writerow([cell.value, "MultiResNet", "Precision"])

for cell in row_daru[1:]:
    i6 += 1
    if i6%6 == 0 :
        j6 += 1
        csv_writer.writerow([cell.value, "DARU-Net", "Precision"])

f.close()

dice_recall = pd.read_csv('data_analysis.csv')

sns.set_style("whitegrid")
ax = sns.boxplot(x="Methods", y="Value", hue="metrics", data=dice_recall, palette="Set3")

#ax = sns.swarmplot(x="Methods", y="Value", hue="metrics", data=dice_recall)
#plt.axhline(y=0.8,ls=":",c="black") #添加水平直线
#plt.grid(ls=":",c='b',) #打开坐标网格

#plt.title('Box plot of DSC and Recall for all test cases')
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
plt.xlabel('')
plt.rc('font',family='Times New Roman')
plt.show()