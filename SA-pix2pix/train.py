import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #放在import torch之前使用，其他文件也尽量加上
import datetime
import numpy as np
import math
import csv
from tqdm import tqdm  # 进度条

from torchvision.utils import save_image  # 用于保存生成的图像

from torch.utils.data import DataLoader  # 用于加载数据集
from torch.autograd import Variable  # 用于自动求导
from torch.optim.lr_scheduler import StepLR  # 用于学习率调度

from models.Generator import *
from models.Discriminator import *


from utils.datasets import *
from utils.SSIM import *
from utils.FeatureExtractor import *

import torch.nn.functional as F  # 提供常用的函数
import torch


print(f'Current Device ID :{torch.cuda.current_device()}') 

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")  # 开始训练的轮数
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")  # 总训练轮数
parser.add_argument("--dataset_org_path", type=str, default="/root/Desktop/work/AAPMSparseGAN/data/AAPMSparseGANdata_org.csv", help="name of the dataset")  ###### 数据集路径
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")  # 每次训练的图片数量
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")  # 学习率
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")  # Adam优化器参数1
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")  # Adam优化器参数2
parser.add_argument("--step_size", type=int, default=100, help="lr decay step size")  # 学习率衰减间隔
parser.add_argument("--img_height", type=int, default=512, help="size of image height")  ###### 图片高度
parser.add_argument("--img_width", type=int, default=512, help="size of image width")  ###### 图片宽度
parser.add_argument("--channels", type=int, default=1, help="number of image channels")  # 图片通道数（这里是灰度图）
parser.add_argument("--checkpoint_interval", type=int, default=100, help="interval between model checkpoints")  # 模型保存间隔
opt = parser.parse_args()
print(opt)

# 创建保存目录
os.makedirs("saved_images", exist_ok=True)  # 保存生成的图片
os.makedirs("saved_models", exist_ok=True)  # 保存训练好的模型
os.makedirs("logs", exist_ok=True)  # 用于存放 PSNR 和 SSIM 的日志目录

cuda = True if torch.cuda.is_available() else False

# Loss functions
criterion = torch.nn.MSELoss()  # 均方误差损失
criterion_pixelwise = torch.nn.L1Loss()  # L1损失函数
criterion_ssim = SSIM(window_size=11)  # 结构相似性损失
criterion_perceptual = PerceptualLoss(layers=15)  # 自定义的感知损失

# 定义判别器的 patch 大小
patch = (1, opt.img_height // 2 ** 5, opt.img_width // 2 ** 5)

# Loss weights 定义损失权重
c = 100
lambda_ = 0.4

# Initialize generator and discriminator 初始化生成器和判别器
gen = GeneratorUNet()
dis = Discriminator()

# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs!")
#     gen = nn.DataParallel(gen)
#     dis = nn.DataParallel(dis)
    
# 如果有多个 GPU，可以使用 nn.DataParallel 进行多 GPU 训练。如果有可用的 GPU，将模型和损失函数移动到 GPU 上。
if cuda:
    generator = gen.cuda()
    discriminator = dis.cuda()
    criterion.cuda()
    criterion_pixelwise.cuda()
    criterion_ssim.cuda()
    criterion_perceptual.cuda()

# Load saved model or initialize weights 加载预训练模型或初始化权重
if opt.epoch != 0:
    pass
    # generator.load_state_dict(torch.load("saved_models/generator_%d.pth" % opt.epoch))
    # discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth" % opt.epoch))
else:
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# Optimizers
# 使用 Adam 优化器优化生成器和判别器的参数
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# 每 step_size 个 epoch 学习率乘以 gamma。将输入的张量归一化到指定的范围
scheduler_G = StepLR(optimizer_G,step_size=opt.step_size,gamma=0.5)
scheduler_D = StepLR(optimizer_D,step_size=opt.step_size,gamma=0.5)
    
# Transforms 数据归一化，把像素值调整到-1到1之间
def transform(tensor, range_min=-1.0, range_max=1.0):
    
    tensor_min = tensor.min()
    tensor_max = tensor.max()

    if tensor_max - tensor_min == 0:
        return tensor

    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min) * (range_max - range_min) + range_min
    return normalized_tensor

# DataLoader 创建数据加载器
train_dataloader = DataLoader(
    AAPMSparseDataset(opt.dataset_org_path, transform_=transform),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=0,
)


def calculate_metrics(gen, high, criterion_ssim):
   
    batch_size = gen.size(0)
    psnr_values = []
    ssim_values = []

    for i in range(batch_size):
        # 分别计算每张图片的 PSNR
        mse = F.mse_loss(gen[i], high[i]).item()
        psnr = 10 * math.log10(1 / mse)
        psnr_values.append(psnr)

        # 分别计算每张图片的 SSIM
        ssim = criterion_ssim(gen[i].unsqueeze(0), high[i].unsqueeze(0)).item()
        ssim_values.append(ssim)

    # 返回 PSNR 和 SSIM 的均值
    psnr_mean = np.mean(psnr_values)
    ssim_mean = np.mean(ssim_values)

    return psnr_mean, ssim_mean


# Tensor type 根据是否使用 GPU，定义张量类型
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Training
for epoch in range(opt.epoch, opt.n_epochs):
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch}/{opt.n_epochs}")
    
    last_gen, last_high = None, None  # 用于保存每个 epoch 最后一个 batch 的生成图像和高分辨率图像

    for i, datapatch in progress_bar:
        # Model inputs
        img_low = Variable(datapatch["L"].type(Tensor))
        img_gt = Variable(datapatch["H"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((img_gt.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((img_gt.size(0), *patch))), requires_grad=False)

        # Train Generators
        optimizer_G.zero_grad()
        img_gen = generator(img_low)

        pred_fake = discriminator(img_gen)

        loss_G_D = criterion(pred_fake, valid)  # 均方误差损失函数 criterion 计算生成器的对抗损失

        loss_pixel = criterion_pixelwise(img_gen, img_gt)  # L1

        loss_ssim = 1 - criterion_ssim(img_gen, img_gt)  # 1-SSIM

        loss_perceptual = criterion_perceptual(img_gen.repeat(1, 3, 1, 1),img_gt.repeat(1, 3, 1, 1))

        # lambda_ = loss_ssim.item() / (loss_ssim.item()+loss_pixel.item())###自适应

        loss_G = loss_G_D + c * (lambda_ * loss_pixel  + (1-lambda_) * loss_ssim ) + loss_perceptual

        loss_G.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        pred_real = discriminator(img_gt)
        loss_real = criterion(pred_real, valid)

        pred_fake = discriminator(img_gen.detach())
        loss_fake = criterion(pred_fake, fake)
        
        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        optimizer_D.step()

        # Update progress bar
        progress_bar.set_postfix({
            "Loss_G": loss_G.item(),
            "Loss_D": loss_D.item(),
            "Loss_SSIM": loss_ssim.item(),
            "Loss_L1": loss_pixel.item(),
            "Loss_P": loss_perceptual.item(),
        })

        # 保存最后一个 batch 的生成结果
        if i == len(train_dataloader) - 1:
            last_gen = img_gen
            last_high = img_gt

    scheduler_G.step()
    scheduler_D.step()


    psnr_mean, ssim_mean = calculate_metrics(last_gen, last_high, criterion_ssim)
 
    # 保存图像
    save_image(last_gen, f"saved_images/epoch_{epoch}_gen.png", nrow=4, normalize=True)
    save_image(last_high, f"saved_images/epoch_{epoch}_groundtruth.png", nrow=4, normalize=True)

    # 保存指标到日志
    with open(f"logs/metrics_+p.csv", "a") as log_file:
        log_writer = csv.writer(log_file)
        if epoch == 0:  # 在第一个 epoch 添加表头
            log_writer.writerow(["Epoch", "PSNR", "SSIM"])
        log_writer.writerow([epoch, psnr_mean, ssim_mean])

    # 保存模型
    if (epoch + 1) % opt.checkpoint_interval == 0:
        torch.save(generator.state_dict(), f"saved_models/generator_+p_{str(datetime.date.today())}_{epoch}.pth")
        # torch.save(discriminator.state_dict(), f"saved_models/discriminator_{str(datetime.date.today())}_{epoch}.pth")

