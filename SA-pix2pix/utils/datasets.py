
import datetime
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import numpy as np
from torchvision.utils import save_image

from torch.utils.data import Dataset


class AAPMSparseDataset(Dataset):
    
    def __init__(self, AAPMSparsedata_org, dtype=np.float32, transform_=None):
        """
        Args:
        - MARdata_org (str): 包含数据路径的 CSV 文件路径。
        - dtype (type): 数据类型。
        - transform (callable, optional): 数据变换方法。
        """
        self.data_info = pd.read_csv(AAPMSparsedata_org)  # 读取 CSV 文件
        self.dtype = dtype
        self.transform = transform_

    def __len__(self):
        """返回数据集的大小"""
        return len(self.data_info)

    def __getitem__(self, idx):
        """
        Args:
        - idx (int): 数据索引。

        Returns:
        - img_medal (Tensor): image with medal。
        - sino_medal (Tensor): sino with medal。
        - img_gt (Tensor): image without medal。
        """
        # 加载 .npy数据
        img_low = np.load(self.data_info.iloc[idx]["img_low"])
        img_gt = np.load(self.data_info.iloc[idx]["img_gt"])

        img_low_tensor = torch.tensor(img_low, dtype=torch.float32).unsqueeze(0)       
        img_gt_tensor = torch.tensor(img_gt, dtype=torch.float32).unsqueeze(0)  

        if self.transform:
            img_low = self.transform(img_low_tensor)
            img_gt = self.transform(img_gt_tensor)

        return {"L": img_low, "H": img_gt}

        
def transforms_(tensor, range_min=-1.0, range_max=1.0):
    
    tensor_min = tensor.min()
    tensor_max = tensor.max()

    if tensor_max - tensor_min == 0:
        return tensor

    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min) * (range_max - range_min) + range_min
    return normalized_tensor

# 示例用法
if __name__ == "__main__":
    # CSV 文件路径
    csv_file = "/root/Desktop/work/AAPMSparseGAN/data/AAPMSparseGANdata_org.csv"  # 替换为实际的 CSV 文件路径

    # 创建 Dataset
    dataset = AAPMSparseDataset(csv_file, dtype=np.float32, transform_=transforms_)

    # 创建 DataLoader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(len(dataloader))

    print(str(datetime.date.today()))
    # 测试数据加载
    for batch_idx, datapatch in enumerate(dataloader):
        img_low,img_gt = datapatch["L"],datapatch["H"]
        print(f"Batch {batch_idx}:")
        print(f"  Input shape: {img_low.shape}")
        print(f"  Target shape: {img_gt.shape}")
        save_image(img_low, "/root/Desktop/work/AAPMSparseGAN/data/img_low1_testdatasets.png", nrow=4, normalize=True)
        save_image(img_gt, "/root/Desktop/work/AAPMSparseGAN/data/img_gt1_testdatasets.png", nrow=4, normalize=True)
        break



