# -*- conding:utf-8 -*-
#  @FileName    :train_test.py
#  @Time        :2020/12/28_17:28
#  @Author      :ZhangJian
#  @Description :

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import random
import pickle
import datetime
import time
from util.ImagePrepared import *
from model.ablation_study.DARU_Net import *

from skimage.io import imsave
from keras import callbacks

def train_and_predict(turn):

    # use .npy to fit
    print('Loading and preprocessing train data...')
    imgs_train, imgs_mask_train = load_train_data()

    imgs_train = imgs_train.astype('float32')
    imgs_train /= 255.  # scale masks to [0, 1]
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization
    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.

    ran = random.randint(0, 999)
    print(ran)
    np.random.seed(ran)
    np.random.shuffle(imgs_train)
    np.random.seed(ran)
    np.random.shuffle(imgs_mask_train)

    imgs_test, imgs_id_test, imgs_test_mask = load_test_data()

    imgs_test = imgs_test.astype('float32')
    imgs_test /= 255.  # scale masks to [0, 1]
    mean = np.mean(imgs_test)
    std = np.std(imgs_test)
    imgs_test -= mean
    imgs_test /= std

    model = get_net()
    model.summary()
    time1 = time.time()

    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_dice', min_delta=0.002, verbose=1, patience=10, mode='max'),
        # These filenames need to be modified in different model training
        callbacks.ModelCheckpoint(os.path.join('data_processed/model_weights', 'Unet_AG{}.h5'.format(turn)),monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1),
        #callbacks.ReduceLROnPlateau(monitor='val_dice', factor=0.9, verbose=1, patience=1, mode='max', min_lr=1e-5)
    ]

    history = model.fit(imgs_train, imgs_mask_train, batch_size=5, epochs=300, verbose=1, shuffle=True,
                        validation_split=0.2, callbacks=callbacks_list)

    # load weights
    model.load_weights(os.path.join('data_processed/model_weights', 'Unet_AG{}.h5'.format(turn)))

    imgs_test_pred = model.predict(imgs_test, batch_size=5,verbose=1)

    print('Predicting masks on test data...')

    # predict to pictures (unit8)
    pred_dir = os.path.join('data_processed', 'Unet_AG{}'.format(turn))
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, id in zip(imgs_test_pred, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, str(id) + '_pred.png'), image)

    time2 = time.time()
    seconds = time1 - time2
    now = datetime.datetime.now()
    print("Time Now: ")
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print("cost time_%02d:%02d:%02d" % (h, m, s))

    # write history to .pickle
    with open('model_Unet_AG{}.pickle'.format(turn), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

# Arrange multiple training - flag:'turn'
train_and_predict(1)
# train_and_predict(2)
# train_and_predict(3)
# train_and_predict(4)