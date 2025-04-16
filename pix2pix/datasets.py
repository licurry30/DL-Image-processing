import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"): #定义时使用
        self.transform = transforms.Compose(transforms_)
        # root 是数据集目录
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        #root/val
        #if mode == "train":
        #    self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))
            #root/train    and  root/test

    def __getitem__(self, index): # 

        img = Image.open(self.files[index % len(self.files)])

        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h)) #crop  与cat对应  0-w/2 左
        img_B = img.crop((w / 2, 0, w, h))

        if np.random.random() < 0.5:
            #img_L = Image.fromarray(np.array(img_L)[:, ::-1, :], "L") #左右翻转   -1代表倒1
            #img_H = Image.fromarray(np.array(img_H)[:, ::-1, :], "L") #翻转   RGB->1 为二值图像
            img_A = Image.fromarray(np.array(img_A)[:, ::-1 ], "RGB") #左右翻转   -1代表倒1
            img_B = Image.fromarray(np.array(img_B)[:, ::-1 ], "RGB") #翻转   RGB->1 为二值图像

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self): #上面调用了这个父类的函数，子类必须声明
        return len(self.files)
