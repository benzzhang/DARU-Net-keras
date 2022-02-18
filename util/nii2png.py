# -*- conding:utf-8 -*-
#  @FileName    :nii2png.py
#  @Time        :2020/11/24_21:19
#  @Author      :ZhangJian
#  @Description : Contains some functions for image operation

import nibabel as nib
import cv2
import numpy as np
import imageio
import os

# convert nii to png #
def nii_to_image():

    # Folder where NII file are stored
    filepath = r'C:\Users\Administrator\Desktop\ss'
    # Folder to save images 'png'
    imgfile = r'C:\Users\Administrator\Desktop\png\1'
    # Folder to save labels 'png'
    labelfile = r'C:\Users\Administrator\Desktop\png\2'

    filenames = os.listdir(filepath)
    # sort
    filenames.sort(key=lambda x: int(x[:-4]))
    k = m = 0
    print(filenames)

    for f in filenames:
        img_path = os.path.join(filepath, f)
        img = nib.load(img_path)
        img_fdata = img.get_fdata()
        # img_fdata = (img.get_fdata()*255).astype(np.uint8)
        fname = int(f.replace('.nii', ''))

        (x, y, z) = img.shape

        if fname % 2 != 0:
            for i in range(z):
                slice = img_fdata[:, :, i]
                imageio.imwrite(os.path.join(imgfile, '{}_{}.png'.format(fname,k)), slice)
                k = k + 1

        else:
            for i in range(z):
                slice = img_fdata[:, :, i]
                imageio.imwrite(os.path.join(labelfile, '{}.png'.format(m)), slice)
                m = m + 1
#nii_to_image()

# convert nii to png after adjusting window width and level #
def nii_to_image_adujust():

    filepath = r'C:\Users\Administrator\Desktop\selected2\test_30cases'
    imgfile = r'C:\Users\Administrator\Desktop\test_image'

    center = 250
    width = 500

    filenames = os.listdir(filepath)
    filenames.sort(key=lambda x: int(x[:-4]))
    k = 0
    print(filenames)

    for f in filenames:
        img_path = os.path.join(filepath, f)
        img = nib.load(img_path)
        img_fdata = img.get_fdata()
        # img_fdata = (img.get_fdata()*255).astype(np.uint8)
        fname = int(f.replace('.nii', ''))

        (x, y, z) = img.shape

        min = (2 * center - width) / 2.0 + 0.5
        max = (2 * center + width) / 2.0 + 0.5
        dFactor = 255.0 / (max - min)

        if fname % 2 != 0:
            for i in range(z):
                slice = img_fdata[:, :, i]

                slice = slice - min
                slice = np.trunc(slice * dFactor)
                slice[slice < 0.0] = 0
                slice[slice > 255.0] = 255  # 转换为窗位窗位之后的数据

                imageio.imwrite(os.path.join(imgfile, '{}.png'.format(k)), slice)
                k = k + 1
#nii_to_image_adujust()

# See how many slices there are in each NII file #
def nii_count(filepath):
    filenames = os.listdir(filepath)
    filenames.sort(key= lambda x:int(x[:-4]))
    s = 0
    print(filenames)
    for f in filenames:
        img_path = os.path.join(filepath, f)
        epi_img = nib.load(img_path)
        width, height, queue = epi_img.dataobj.shape
        print(img_path +'__'+ str(queue)+ 'slices')
        s = s + queue
    print('total'+'_'+str(s)+'slices')


# Find the picture whose size is not 512 #
def seleted_not_512(filepath):
    # filepath: 保存原图的文件夹 -PNG
    img = os.listdir(filepath)
    for i in img:
        # print(i)
        k = os.path.join(filepath, i)
        # print(k)
        size = cv2.imread(k)
        # print(size.shape[1])
        if size.shape[1] != 512:
            print(k)


# View grayscale values
def show_the_Gray():
    img = cv2.imread(r'C:\Users\Administrator\Desktop\22.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    def mouse_click(event, x, y, flags, para):
        if event == cv2.EVENT_LBUTTONDOWN:  # left button down MOUSE
            print('PIX:', x, y)
            print("BGR:", img[y, x])
            print("GRAY:", gray[y, x])
            print("HSV:", hsv[y, x])


    cv2.namedWindow("img")
    cv2.setMouseCallback("img", mouse_click)
    while True:
        cv2.imshow('img', img)
        if cv2.waitKey() == ord('q'):
            break
    cv2.destroyAllWindows()
#show_the_Gray()

# Rename #
def Rename():
    path = r'C:\Users\Administrator\Desktop\2222'
    path0 = r'C:\Users\Administrator\Desktop\3333'

    file = os.listdir(path)
    #file.sort(key=lambda x: int(x[:-4]))
    print(file)
    m = 0

    for i in range(len(file)):
        os.rename(os.path.join(path, file[i]), os.path.join(path0, '{}.nii'.format(i+115)))
    '''
    for i in range(len(file)):
        if i % 2 != 0:
            os.rename(os.path.join(path, file[i]), os.path.join(path0, '{}.nii'.format(i+150)))
        else:
            os.rename(os.path.join(path, file[i]), os.path.join(path0, '{}.nii'.format(i+152)))
    '''
#Rename()

# Delete the pictures not in the 'path_label' from 'path_image' #
def Delete_Img():
    path_image = r'C:\Users\Administrator\Desktop\selected_image\train\8比2image_adjust'
    path_label = r'C:\Users\Administrator\Desktop\selected_image\train\8比2mask'

    file_image = os.listdir(path_image)

    for i in file_image:
        if (not os.path.exists(os.path.join(path_label, i))):
            os.remove(os.path.join(path_image, i))
#Delete_Img()

# Identify duplicate cases #
def duplicate():

    path1 = r'C:\Users\Administrator\Desktop\111'
    path2 = r'C:\Users\Administrator\Desktop\222'

    filenames = os.listdir(path1)
    filenames.sort(key=lambda x: int(x[:-6]))

    filenames2 = os.listdir(path2)
    filenames2.sort(key=lambda x: int(x[:-4]))

    print(filenames)
    number = []
    for f2 in filenames2:
        img_path2 = os.path.join(path2, f2)
        img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)

        for f1 in filenames:
            img_path = os.path.join(path1, f1)
            img1 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if((img1==img2).all()):
                number.append(f1[:-6])
    number.sort(key=lambda x: int(x))
    print(number)
#duplicate()

# Output image with case number #
def nii_to_image_with_Number():

    filepath = r'F:\fibroid_data\recieved_from_Zheng_first\HT2'
    imgfile = r'C:\Users\Administrator\Desktop\111'

    filenames = os.listdir(filepath)
    filenames.sort(key=lambda x: int(x[:-4]))
    k = 0
    print(filenames)

    for f in filenames:
        img_path = os.path.join(filepath, f)
        img = nib.load(img_path)
        img_fdata = img.get_fdata()
        # img_fdata = (img.get_fdata()*255).astype(np.uint8)
        fname = int(f.replace('.nii', ''))

        (x, y, z) = img.shape

        if fname % 2 != 0:
            for i in range(z):
                slice = img_fdata[:, :, i]
                imageio.imwrite(os.path.join(imgfile, '{0}_{1}.png'.format(fname,k)), slice)
                #k = k + 1
                break
#nii_to_image_with_Number()

# Delete pictures without targets (WHITE AREA) #
def delete():
    path =r'C:\Users\Administrator\Desktop\Label'
    pics = os.listdir(path)
    for pic in pics:
        de = os.path.join(path,pic)
        print(de)
        image = cv2.imread(de,cv2.IMREAD_GRAYSCALE)
        if image.any():
            print(0)
        else:
            os.remove(de)
