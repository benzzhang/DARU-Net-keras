# -*- conding:utf-8 -*-
#  @FileName    :MultiResUNet.py
#  @Time        :2020/11/30_19:05
#  @Author      :ZhangJian
#  @Description : Paper url: https://arxiv.org/abs/1902.04049

from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, \
    Activation, add, BatchNormalization
from keras.models import Model
from Loss_Metrics.Loss import *
from keras.optimizers import Adam

def conv2d_B_A(input,filters,size):
    x = Conv2D(filters,kernel_size=(size,size), padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

def Mutil_Res_block(input, filter_size1, filter_size2, filter_size3, filter_size4):
    cnn3x3 = conv2d_B_A(input, filter_size1, 3)
    cnn5x5 = conv2d_B_A(cnn3x3, filter_size2, 3)
    cnn7x7 = conv2d_B_A(cnn5x5, filter_size3, 3)

    shortcut = conv2d_B_A(input, filter_size4, 1)

    concat = concatenate([cnn3x3,cnn5x5,cnn7x7])
    ad = add([shortcut, concat])

    return ad

def Res_Path(input, filter_size, path_num):
    def block(x, f1):

        cnn1 = conv2d_B_A(x, f1, 3)
        cnn2 = conv2d_B_A(x, f1, 1)

        ad = add([cnn1, cnn2])

        return ad

    cnn = block(input, filter_size)
    if path_num <= 3:
        cnn = block(cnn, filter_size)
        if path_num <= 2:
            cnn = block(cnn, filter_size)
            if path_num <= 1:
                cnn = block(cnn, filter_size)

    return cnn

def UpSampling(filter, input):
    x = UpSampling2D(size=(2, 2))(input)
    x = Conv2D(filter, (3, 3), activation='relu', padding='same')(x)

    return x

def get_net():
    inputs = Input((512,512,1))

    res_block1 = Mutil_Res_block(inputs, 8,17,26,51)
    pool1 = MaxPooling2D()(res_block1)

    res_block2 = Mutil_Res_block(pool1, 17,35,53,105)
    pool2 = MaxPooling2D()(res_block2)

    res_block3 = Mutil_Res_block(pool2, 31,72,106,209)
    pool3 = MaxPooling2D()(res_block3)

    res_block4 = Mutil_Res_block(pool3, 71,142,213,426)
    pool4 = MaxPooling2D()(res_block4)

    res_block5 = Mutil_Res_block(pool4, 142,284,427,853)
    upsampling = UpSampling2D()(res_block5)

    res_path4 = Res_Path(res_block4, 256, 4)
    concat = concatenate([upsampling, res_path4])

    res_block6 = Mutil_Res_block(concat, 71,142,213,426)
    upsampling = UpSampling2D()(res_block6)

    res_path3 = Res_Path(res_block3, 128, 3)
    concat = concatenate([upsampling, res_path3])

    res_block7 = Mutil_Res_block(concat, 31, 72, 106, 209)
    upsampling = UpSampling2D()(res_block7)

    res_path2 = Res_Path(res_block2, 64, 2)
    concat = concatenate([upsampling, res_path2])

    res_block8 = Mutil_Res_block(concat, 17, 35, 53, 105)
    upsampling = UpSampling2D()(res_block8)

    res_path1 = Res_Path(res_block1, 32, 1)
    concat = concatenate([upsampling, res_path1])

    res_block9 = Mutil_Res_block(concat, 8,17,26,51)
    sigmoid = Conv2D(1, (1,1), padding='same', activation='sigmoid')(res_block9)

    model = Model(inputs, sigmoid)
    model.compile(optimizer=Adam(lr=2e-4), loss=dice_loss, metrics=[dice])

    return model