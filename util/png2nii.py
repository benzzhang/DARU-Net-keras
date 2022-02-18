# -*- conding:utf-8 -*-
#  @FileName    :png2nii.py
#  @Time        :2021/5/27_10:33
#  @Author      :ZhangJian
#  @Description : .png -> .nii

from PIL import Image
import numpy as np
import SimpleITK as sitk
import os

path = r'C:\Users\Administrator\Desktop\empty'
out_path= r'C:\Users\Administrator\Desktop\pred'
file_name = "pred.nii.gz"

for case in os.listdir(path):

    png_path = os.path.join(path, case)
    name = str(case)+file_name
    nii_path = os.path.join(out_path, name)
    files = os.listdir(png_path)
    files.sort(key=lambda x: int(x[:-9]))

    empt_mat=[]
    for i in files:
        img1=Image.open(os.path.join(png_path,i))
        img2=np.array(img1)
    # take the first three channels of PNG image and remove the fourth transparent channel
        empt_mat.append(img2)

    emp=np.array(empt_mat)
    nii_file = sitk.GetImageFromArray(emp)
    # emp: images_num * height * width * channels

    sitk.WriteImage(nii_file,nii_path)
