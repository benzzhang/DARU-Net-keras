# -*- conding:utf-8 -*-
#  @FileName    :DataPreparation.py
#  @Time        :2020/11/16_11:06
#  @Author      :ZhangJian
#  @Description : Prepare data for training and testing

import os
import numpy as np

# skimage.io read data -- numpy
from skimage.io import imread

# Folder to save training and test data
data_path = 'data'
# Folder to save processed data
data_processed_path = 'data_processed'

# the size of image
image_rows = 512
image_cols = 512

def create_train_data():

    train_data_path = os.path.join(data_path, 'train/image')
    train_data_Label_path = os.path.join(data_path, 'train/label')
    images = os.listdir(train_data_path)
    total = len(images)

    # create empty Array to save training data (value: 0-255)
    imgs = np.ndarray((total, image_rows, image_cols, 1), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols, 1), dtype=np.uint8)

    i = 0
    print('-' * 30)
    print('Creating training images...')
    print('-' * 30)
    for image_name in images:
        # Read images and labels
        img = imread(os.path.join(train_data_path, image_name), as_gray=True)
        img_mask = imread(os.path.join(train_data_Label_path, image_name), as_gray=True)

        # Convert to array
        img = np.array([img])
        img = img[:,:,:,np.newaxis]
        img_mask = np.array([img_mask])
        img_mask = img_mask[:,:,:,np.newaxis]

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    # Save as .NPY file (special binary file for storing arrays in numpy)
    np.save(os.path.join(data_processed_path, 'imgs_train.npy'), imgs)
    np.save(os.path.join(data_processed_path, 'imgs_train_mask.npy'), imgs_mask)
    print('Saving to .npy files done.')

def load_train_data():
    imgs_train = np.load(os.path.join(data_processed_path, 'imgs_train.npy'))
    imgs_mask_train = np.load(os.path.join(data_processed_path, 'imgs_train_mask.npy'))
    return imgs_train, imgs_mask_train

def create_test_data():

    test_data_path = os.path.join(data_path, 'test/image')
    images = os.listdir(test_data_path)
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols, 1), dtype=np.uint8)
    imgs_id = np.ndarray((total,), dtype=np.int32)

    i = 0
    print('-' * 30)
    print('Creating test images...')
    print('-' * 30)
    for image_name in images:
        #  split the file name with '.' and take the id
        img_id = int(image_name.split('.')[0])
        img = imread(os.path.join(test_data_path, image_name), as_gray=True)

        img = np.array([img])
        img = img[:,:,:,np.newaxis]

        imgs[i] = img

        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(os.path.join(data_processed_path, 'imgs_test.npy'), imgs)
    np.save(os.path.join(data_processed_path, 'imgs_test_id.npy'), imgs_id)
    print('Saving to .npy files done.')

def load_test_data():
    imgs_test = np.load(os.path.join(data_processed_path, 'imgs_test.npy'))
    imgs_id = np.load(os.path.join(data_processed_path, 'imgs_test_id.npy'))
    return imgs_test, imgs_id

create_train_data()
create_test_data()