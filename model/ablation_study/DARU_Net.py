# -*- conding:utf-8 -*-
#  @FileName    :DARU_Net.py
#  @Time        :2021/7/15_10:03
#  @Author      :ZhangJian
#  @Description :

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, Reshape, add,\
    MaxPooling2D, UpSampling2D, Activation, AveragePooling2D, \
    BatchNormalization, GlobalAveragePooling2D, Dense, Multiply
from Loss_Metrics.Loss import *
from keras.optimizers import Adam

def PPM_Block(input, input_size):
    x = input
    print(K.int_shape(x))
    shapex = K.int_shape(x)[3]
    # Multi-kernel pooling
    pool_1x1 = AveragePooling2D((input_size//4, input_size//4), strides=(input_size//4, input_size//4))(input)
    pool_2x2 = AveragePooling2D((input_size//8, input_size//8), strides=(input_size//8, input_size//8))(input)
    pool_4x4 = AveragePooling2D((input_size//16, input_size//16), strides=(input_size//16, input_size//16))(input)
    pool_8x8 = AveragePooling2D((input_size//32, input_size//32), strides=(input_size//32, input_size//32))(input)
    print(str('#'), pool_1x1.shape, pool_2x2.shape, pool_4x4.shape, pool_8x8.shape)

    # Use 1x1 Convolution to reduce the dimension of the feature maps to the 1/N of original dimension.
    pool_1x1_Conved = Conv2D(shapex // 4, (1, 1), activation='relu', padding='same')(pool_1x1)
    pool_2x2_Conved = Conv2D(shapex // 4, (1, 1), activation='relu', padding='same')(pool_2x2)
    pool_4x4_Conved = Conv2D(shapex // 4, (1, 1), activation='relu', padding='same')(pool_4x4)
    pool_8x8_Conved = Conv2D(shapex // 4, (1, 1), activation='relu', padding='same')(pool_8x8)
    print(str('#'), pool_1x1_Conved.shape, pool_2x2_Conved.shape, pool_4x4_Conved.shape, pool_8x8_Conved.shape)

    # Upsampling the low-dimension feature map toget the same size features as the original feature map via bilinear interpolation.
    up_1x1 = UpSampling2D(size=(input_size//4, input_size//4))(pool_1x1_Conved)
    up_2x2 = UpSampling2D(size=(input_size//8, input_size//8))(pool_2x2_Conved)
    up_4x4 = UpSampling2D(size=(input_size//16, input_size//16))(pool_4x4_Conved)
    up_8x8 = UpSampling2D(size=(input_size//32, input_size//32))(pool_8x8_Conved)
    print(str('#'), up_1x1.shape, up_2x2.shape, up_4x4.shape, up_8x8.shape)

    PPM_result = concatenate([input, up_1x1, up_2x2, up_4x4, up_8x8], axis=-1)

    return PPM_result

def SE_Block(input, ratio):

    channel = int(input.shape[-1])
    print('SE输入channels: %d'%channel)
    # 1*1*C
    squeeze = GlobalAveragePooling2D()(input)

    excitation = Dense(units=channel//ratio)(squeeze)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=channel)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape((1,1,channel))(excitation)

    scale = Multiply()([input, excitation])

    return  scale

def conv2d_layer(input, channles):

    conv1 = Conv2D(channles, 3, use_bias=False, padding='same')(input)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    conv2 = Conv2D(channles, 3, use_bias=False, padding='same')(conv1)

    conv = SE_Block(conv2, 8)

    res_sum = add([conv, conv1])
    res_sum = BatchNormalization()(res_sum)
    res_sum = Activation('relu')(res_sum)
    return res_sum

def get_net():
    inputs = Input((512, 512, 1))

    conv1 = conv2d_layer(inputs, 32)
    pool1_1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1_2 = AveragePooling2D(pool_size=(2, 2))(conv1)
    pool1 = add([pool1_1, pool1_2])

    conv2 = conv2d_layer(pool1, 64)
    pool2_1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2_2 = AveragePooling2D(pool_size=(2, 2))(conv2)
    pool2 = add([pool2_1, pool2_2])

    # 128*128
    conv3 = conv2d_layer(pool2, 128)
    pool3_1 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3_2 = AveragePooling2D(pool_size=(2, 2))(conv3)
    pool3 = add([pool3_1, pool3_2])

    # 64*256
    conv4 = conv2d_layer(pool3, 256)
    pool4_1 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4_2 = AveragePooling2D(pool_size=(2, 2))(conv4)
    pool4 = add([pool4_1, pool4_2])

    # 32*512
    conv5 = conv2d_layer(pool4, 512)
    conv5 = PPM_Block(conv5, 32)
    print("conv5:",conv5.shape)

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

get_net()

