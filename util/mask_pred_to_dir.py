# -*- conding:utf-8 -*-
#  @FileName    :mask_pred_to_dir.py
#  @Time        :2021/3/30_10:15
#  @Author      :ZhangJian
#  @Description :

import nibabel as nib
import os
import shutil

def mkdirr():
    # 根据测试集nii文件，为每个病例生成一个文件夹

    filepath = r'C:\Users\Administrator\Desktop\selected2\test_30cases'
    dstpath = r'C:\Users\Administrator\Desktop\selected_image\each_mask'
    filenames = os.listdir(filepath)
    filenames.sort(key=lambda x: int(x[:-4]))
    print(filenames)

    for f in filenames:
        fname = int(f.replace('.nii', ''))
        if fname % 2 == 0:
            new = os.path.join(dstpath,str(fname))
            os.mkdir(new)

#mkdirr()

def move_pics():
    # 将测试集的分割结果’按每个病例‘，分到1-->N的文件夹内，用以之后计算 dice precision recall...

    # segmentation results after threshold #
    filepath = r'C:\Users\Administrator\Desktop\222'
    fx = os.listdir(filepath)
    fx.sort(key=lambda x:int(x[:-9]))
    print(fx[0])
    #获得每张Mask地址

    # 导出保存位置 #
    dstpath = r'C:\Users\Administrator\Desktop\empty'
    dst = os.listdir(dstpath)
    dst.sort(key=lambda x:int(x))
    print(dst[0])
    #获得保存到的目标地址

    countpath = r'C:\Users\Administrator\Desktop\selected2\test_30cases'
    filenames = os.listdir(countpath)
    filenames.sort(key=lambda x: int(x[:-4]))
    m = 0  # 计数器

    start = 0
    for f in filenames:
        fname = int(f.replace('.nii', ''))
        print(fname,'ss')
        if fname % 2 ==0:
            img_path = os.path.join(countpath, f)
            img = nib.load(img_path)
            (x, y, z) = img.shape
            # 获得了每个Mask文件的张数

            end = start+z
            print(start,'~',end)
            for i in range(start, end):
                shutil.copy(os.path.join(filepath,fx[i]), os.path.join(dstpath, dst[m]))
            start = end
            m = m + 1

#move_pics()