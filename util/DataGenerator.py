# -*- conding:utf-8 -*-
#  @FileName    :DataGenerator.py
#  @Time        :2020/12/1_20:25
#  @Author      :ZhangJian
#  @Description : Data Augmentation

from keras.preprocessing.image import ImageDataGenerator

Label_ori_path = r'C:\Users\Administrator\Desktop\2'
Label_dst_path = r'C:\Users\Administrator\Desktop\selected_image\train'
Image_ori_path = r'C:\Users\Administrator\Desktop\1'
Image_dst_path = r'C:\Users\Administrator\Desktop\selected_image\train'

data_gen_args = dict(
    rotation_range=15,       # 0-180
    width_shift_range=0.1,   # 0-1
    height_shift_range=0.1,  # 0-1
    #shear_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'      # Pixel fill mode
)

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

seed = 2

image_generator = image_datagen.flow_from_directory(
    Image_ori_path,
    save_to_dir=Image_dst_path,
    batch_size=50,
    target_size=(512,512),
    class_mode=None,
    seed=seed,
    color_mode='grayscale'
)
mask_generator = mask_datagen.flow_from_directory(
    Label_ori_path,
    save_to_dir=Label_dst_path,
    batch_size=50,
    target_size=(512, 512),
    class_mode=None,
    seed=seed,
    color_mode='grayscale'
)

for i in range(100):
    image_generator.next()
    mask_generator.next()
