# -*- conding:utf-8 -*-
#  @FileName    :UNet_Res.py
#  @Time        :2021/7/11_14:23
#  @Author      :ZhangJian
#  @Description :

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, add, \
    MaxPooling2D, UpSampling2D, Activation, BatchNormalization
from Loss_Metrics.Loss import *
from keras.optimizers import Adam


def conv2d_layer(input, channles):

    conv1 = Conv2D(channles, 3, use_bias=False, padding='same')(input)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    conv2 = Conv2D(channles, 3, use_bias=False, padding='same')(conv1)

    res_sum = add([conv2, conv1])
    res_sum = BatchNormalization()(res_sum)
    res_sum = Activation('relu')(res_sum)
    return res_sum

def get_net():
    inputs = Input((512, 512, 1))

    conv1 = conv2d_layer(inputs, 32)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv2d_layer(pool1, 64)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # 128*128
    conv3 = conv2d_layer(pool2, 128)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # 64*256
    conv4 = conv2d_layer(pool3, 256)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # 32*512
    conv5 = conv2d_layer(pool4, 512)

    up6 = concatenate(
        [Conv2D(256, 3, activation='relu', padding='same')(UpSampling2D(size=(2, 2), interpolation='bilinear')(conv5)),
         conv4], axis=-1)
    conv6 = conv2d_layer(up6, 256)

    up7 = concatenate(
        [Conv2D(128, 3, activation='relu', padding='same')(UpSampling2D(size=(2, 2), interpolation='bilinear')(conv6)),
         conv3], axis=-1)
    conv7 = conv2d_layer(up7, 128)

    up8 = concatenate(
        [Conv2D(64, 3, activation='relu', padding='same')(UpSampling2D(size=(2, 2), interpolation='bilinear')(conv7)),
         conv2], axis=-1)
    conv8 = conv2d_layer(up8, 64)

    up9 = concatenate(
        [Conv2D(32, 3, activation='relu', padding='same')(UpSampling2D(size=(2, 2), interpolation='bilinear')(conv8)),
         conv1], axis=-1)
    conv9 = conv2d_layer(up9, 32)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer=Adam(lr=2e-4), loss=dice_loss, metrics=[dice])

    return model




