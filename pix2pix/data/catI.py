from PIL import Image
import os
import cv2
import numpy as np

# 两个文件夹的路径
dir_low = "/root/Desktop/work/work-04-01/Pix2Pix/data/training_label/"
dir_high = "/root/Desktop/work/work-04-01/Pix2Pix/data/training_data/"



# 循环遍历两个文件夹下的所有同名图片，进行拼接
for file_name in os.listdir(dir_low):
    if file_name.endswith(".png"):
        # 读取两个文件夹下的同名图片
        #cv2的imread默认读取三通道，加参数使得读取图像为单通道灰度图
        img_low = cv2.imread(dir_low + file_name) 
        img_high = cv2.imread(dir_high + file_name)       
        # 拼接图片
        print("doing... ", dir_high+file_name)
        im_new = np.concatenate((img_low, img_high), axis=1)  # 横向拼接
        # 显示图片（数组到图片转换）
        im_new   = Image.fromarray(im_new,"RGB")

        # 保存拼接后的图片
        im_new.save("/root/Desktop/work/work-04-01/Pix2Pix/data/custom/train/" + file_name) #
