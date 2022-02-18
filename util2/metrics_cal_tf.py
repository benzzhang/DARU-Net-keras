# -*- conding:utf-8 -*-
#  @Description : tf计算dsc, precision, recall

import nibabel as nib
import scipy.io as io
import os
import numpy as np
import tensorflow as tf
from keras import backend as K
from scipy.io import loadmat

tf.compat.v1.disable_eager_execution()

def dice_coefficient(y_true, y_pred, smooth=0.00001):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice / 255

def precision(y_true, y_pred, smooth=0.00001):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    pre = (intersection + smooth) / (K.sum(y_pred_f) + smooth)
    return pre / 255

def recall(y_true, y_pred, smooth=0.00001):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    Recall = (intersection + smooth) / (K.sum(y_true_f) + smooth)
    return Recall / 255

def Jaccard_Index(y_true, y_pred, smooth=0.00001):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    JI = (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)
    return JI

pred_dir = r'C:\Users\Administrator\Desktop\pred'
true_dir = r'C:\Users\Administrator\Desktop\true'
pred_filenames = os.listdir(pred_dir)
pred_filenames.sort(key=lambda x: x[:-11])

true_filenames = os.listdir(true_dir)
true_filenames.sort(key=lambda x: x[:-11])

dice_value = np.zeros(30)
temp_dice = []
temp_precision = []
temp_recall = []
temp_JI = []

for f,s in zip(pred_filenames,true_filenames):
    pred_path = os.path.join(pred_dir, f)
    #print(pred_path)
    img_pred = nib.load(pred_path)
    y_pred = img_pred.get_fdata()
    true_path = os.path.join(true_dir, s)
    #print(true_path)
    img_true = nib.load(true_path)
    y_true = img_true.get_fdata()
    temp_dice.append(dice_coefficient(y_true, y_pred))
    temp_precision.append(precision(y_true, y_pred))
    temp_recall.append(recall(y_true, y_pred))
    #temp_JI.append(Jaccard_Index(y_true, y_pred))
    pass

with tf.compat.v1.Session() as sess:
    dice_value = sess.run(temp_dice)
    precision_value = sess.run(temp_precision)
    recall_value = sess.run(temp_recall)
    #JI_value = sess.run(temp_JI)
    pass

save_path = r'C:\Users\Administrator\Desktop\nice'
file_name = 'value.mat'

mat_path = os.path.join(save_path, file_name)
io.savemat(mat_path, {'DICE_value': dice_value, 'Precision_value': precision_value,
                      'Recall_value': recall_value})

features_struct=loadmat(mat_path)

print(features_struct['DICE_value'])
print('Dice:', np.mean(features_struct['DICE_value']))

print(features_struct['Precision_value'])
print('Precision:', np.mean(features_struct['Precision_value']))

print(features_struct['Recall_value'])
print('Recall:', np.mean(features_struct['Recall_value']))
