# -*- conding:utf-8 -*-
#  @FileName    :Attention_UNet.py
#  @Time        :2021/7/10_19:58
#  @Author      :ZhangJian
#  @Description : Paper url: https://arxiv.org/abs/1804.03999

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Activation, BatchNormalization, Add, Multiply
from Loss_Metrics.Loss import *
from keras.optimizers import Adam

def conv2d_layer(input, channles):

    conv1 = Conv2D(channles, 3, use_bias=False, padding='same')(input)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    conv2 = Conv2D(channles, 3, use_bias=False, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)

    return conv2

def Attention_Gate(input_g, input_x, filter):

    # x是浅层的下采样路径中的feature，
    # g是深层的上采样路径中的feature

    # 用上采样后的g(门控信号)来指导x形成权重矩阵

    g1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(input_g)
    g1 = Conv2D(filter, (1, 1))(g1)

    x1 = Conv2D(filter, (1, 1))(input_x)

    psi = Add()([g1, x1])
    psi = Activation('relu')(psi)

    psi = Conv2D(1, (1, 1))(psi)
    psi = BatchNormalization()(psi)
    psi = Activation('sigmoid')(psi)
    out = Multiply()([input_x, psi])
    return out

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

    atten4 = Attention_Gate(conv5, conv4, 256)
    up6 = concatenate(
        [Conv2D(256, 3, activation='relu', padding='same')(UpSampling2D(size=(2, 2), interpolation='bilinear')(conv5)),
         atten4], axis=-1)
    conv6 = conv2d_layer(up6, 256)

    atten3 = Attention_Gate(conv6, conv3, 256)
    up7 = concatenate(
        [Conv2D(128, 3, activation='relu', padding='same')(UpSampling2D(size=(2, 2), interpolation='bilinear')(conv6)),
         atten3], axis=-1)
    conv7 = conv2d_layer(up7, 128)

    atten2 = Attention_Gate(conv7, conv2, 256)
    up8 = concatenate(
        [Conv2D(64, 3, activation='relu', padding='same')(UpSampling2D(size=(2, 2), interpolation='bilinear')(conv7)),
         atten2], axis=-1)
    conv8 = conv2d_layer(up8, 64)

    up9 = concatenate(
        [Conv2D(32, 3, activation='relu', padding='same')(UpSampling2D(size=(2, 2), interpolation='bilinear')(conv8)),
         conv1], axis=-1)
    conv9 = conv2d_layer(up9, 32)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer=Adam(lr=2e-4), loss=dice_loss, metrics=[dice])

    return model




