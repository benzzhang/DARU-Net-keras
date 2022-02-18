# -*- conding:utf-8 -*-
#  @FileName    :read_speed.py
#  @Time        :2021/9/26_12:51
#  @Author      :ZhangJian
#  @Description : shows the evolution of DSC for the training and validation set

import matplotlib.pyplot as plt
import numpy as np
import pickle

plt.rc('font',family='Times New Roman')

with open('model_net_U200.pickle', 'rb') as f:
    history_u = pickle.load(f)

with open('model_net_AG200.pickle', 'rb') as f:
    history_ag = pickle.load(f)

with open('model_net_LED200.pickle', 'rb') as f:
    history_led = pickle.load(f)

with open('model_net_HR200.pickle', 'rb') as f:
    history_hr = pickle.load(f)

with open('model_net_Mul200.pickle', 'rb') as f:
    history_mul = pickle.load(f)

with open('model_net_proposed200.pickle', 'rb') as f:
    history_proposed = pickle.load(f)

# print(history_u.keys())
# print(history_u)
# print(history_ag)
# print(history_hr)
# print(history_u['Loss_Metrics'])
print(history_proposed['dice'])

maxlen = max(len(history_u['Loss_Metrics']), len(history_ag['Loss_Metrics'])
             , len(history_hr['Loss_Metrics']), len(history_led['Loss_Metrics'])
                   ,len(history_mul['Loss_Metrics']), len(history_proposed['Loss_Metrics']))
my_x_ticks = np.arange(1, maxlen, 50)
my_y_ticks = np.arange(0, 1, 0.1)

plt.subplot(1,2,1)

plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)

plt.plot(history_u['dice'][0:200],lw=1)
plt.plot(history_ag['dice'],lw=1)
plt.plot(history_led['dice'],lw=1)
plt.plot(history_hr['dice'],lw=1)
plt.plot(history_mul['dice'][0:200],lw=1)
plt.plot(history_proposed['dice'],lw=1)

plt.axhline(0.8, color='gray', linestyle='--', lw=1)
plt.title('(a) Evolution of DSC in the training set')
plt.ylabel('DSC')
plt.xlabel('Epochs')
plt.legend(['U-Net', 'Attention U-Net', 'LEDNet', 'HRNet', 'MultiResNet', 'DARU-Net'], loc='best')


plt.subplot(1,2,2)

plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)

plt.plot(history_u['val_dice'][0:200],lw=1)
plt.plot(history_ag['val_dice'],lw=1)
plt.plot(history_led['val_dice'],lw=1)
plt.plot(history_hr['val_dice'],lw=1)
plt.plot(history_mul['val_dice'][0:200],lw=1)
plt.plot(history_proposed['val_dice'],lw=1)

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
plt.axhline(0.8, color='gray', linestyle='--', lw=1)
plt.title('(b) Evolution of DSC in the validation set')
plt.ylabel('DSC')
plt.xlabel('Epochs')
plt.legend(['U-Net', 'Attention U-Net', 'LEDNet', 'HRNet', 'MultiResNet', 'DARU-Net'], loc='best')
plt.show()
