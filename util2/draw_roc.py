# -*- conding:utf-8 -*-
#  @FileName    :DrawROC.py
#  @Time        :2022/3/24_12:07
#  @Author      :ZhangJian
#  @Description :

from sklearn.metrics import auc
import matplotlib.pyplot as plt
from matplotlib import rcParams
config = {
    "font.family":'Arial',
}
rcParams.update(config)

fpr_u = [0,0.0011,1]
tpr_u = [0,0.7187,1]

fpr_ag = [0,0.0008,1]
tpr_ag = [0,0.7536,1]

fpr_led = [0,0.0006,1]
tpr_led = [0,0.6516,1]

fpr_hr = [0,0.0012,1]
tpr_hr = [0,0.7787,1]

fpr_mul = [0,0.0007,1]
tpr_mul = [0,0.7255,1]

fpr_daru = [0,0.0009,1]
tpr_daru =  [0,0.7985,1]

roc_auc_u = auc(fpr_u, tpr_u)
roc_auc_ag = auc(fpr_ag, tpr_ag)
roc_auc_led = auc(fpr_led, tpr_led)
roc_auc_hr = auc(fpr_hr, tpr_hr)
roc_auc_mul = auc(fpr_mul, tpr_mul)
roc_auc_daru = auc(fpr_daru, tpr_daru)


plt.plot(fpr_u, tpr_u, 'k--', color='#f15a22', label='ROC_U-Net (area = {0:.2f})'.format(roc_auc_u), lw=2)
plt.plot(fpr_ag, tpr_ag, 'k--', color='#000000', label='ROC_Attention U-Net (area = {0:.2f})'.format(roc_auc_ag), lw=2)
plt.plot(fpr_led, tpr_led, 'k--', color='#ba8448', label='ROC_LEDNet (area = {0:.2f})'.format(roc_auc_led), lw=2)
plt.plot(fpr_hr, tpr_hr, 'k--', color='#694d9f', label='ROC_HRNet (area = {0:.2f})'.format(roc_auc_hr), lw=2)
plt.plot(fpr_mul, tpr_mul, 'k--', color='#1d953f', label='ROC_MultiResNet (area = {0:.2f})'.format(roc_auc_mul), lw=2)
plt.plot(fpr_daru, tpr_daru, 'k--', color='#ea6785', label='ROC_DARU-Net (area = {0:.2f})'.format(roc_auc_daru), lw=2)


plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像
plt.ylim([-0.05, 1.05])
font1 = {'size':10}
plt.xlabel('False Positive Rate',font1)
plt.ylabel('True Positive Rate',font1)
plt.title('ROC Curve')
plt.legend(loc="lower right")
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_color('gray')
ax.spines['top'].set_color('gray')
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
plt.show()