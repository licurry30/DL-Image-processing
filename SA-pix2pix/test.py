import numpy as np
import argparse
import torch
import torch.nn as nn
from PIL import Image
from model_unet_sa import GeneratorUNet
import os
import time
import csv
import math
from skimage.metrics import structural_similarity as ssim
import tracemalloc

os.environ['CUDA_VISIBLE_DEVICES'] ='' # 禁用显卡

class Recon(object):

    def __init__(self,args):
        
        self.RDNet = self.load_model(args.model_path)
        self.epoch = args.epoch
        self.testimgs_folder = args.testimgs_folder
        self.results_folder = args.results_folder

    # 加载模型
    def load_model(self,model_path):

        model = GeneratorUNet().to('cpu')
        model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
        return model


    def transforms_(self,tensor, range_min=-1.0, range_max=1.0):
        
        tensor_min = tensor.min()
        tensor_max = tensor.max()

        if tensor_max - tensor_min == 0:
            return tensor

        normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min) * (range_max - range_min) + range_min
        return normalized_tensor


    def model_process(self, img_array, model):

        pixel_tensor = self.transforms_(torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        recon_tensor = model(pixel_tensor)

        recon_tensor = torch.squeeze(recon_tensor)
        recon_np = recon_tensor.detach().numpy()

        recon_np = (np.float32(recon_np) - np.min(recon_np))/ (np.max(recon_np) - np.min(recon_np))
        return recon_np
        

    def calculate_metrics(self, gen, high):

        gen = (np.float32(gen) - np.min(gen))/ (np.max(gen) - np.min(gen))
        high = (np.float32(high) - np.min(high))/ (np.max(high) - np.min(high))

        mse = np.mean((gen-high) ** 2)
        psnr_index = 10 * math.log10(1 / mse)

        ssim_index, _ = ssim(gen, high, data_range=1.0,full=True)

        return psnr_index, ssim_index

        
    def run(self):

        lowimgfolder = self.testimgs_folder + 'lowimgs/'
        filenum = len(os.listdir(lowimgfolder))
        print('共有',filenum,'个文件。')

        tracemalloc.start()  
        for i in range(filenum):
        
            start_time = time.time()
            lowimg = np.load(lowimgfolder + f'lowimgname_{i+1}.npy') ###
            gtimg = np.load(self.testimgs_folder + f'gtimgs/gtimgname_{i+1}.npy')
 
            #
            genimg = self.model_process(lowimg,self.RDNet)
            #
            psnr_value, ssim_value = self.calculate_metrics(genimg, gtimg)
            #
            with open(self.results_folder + f"metrics_epoch={self.epoch}.csv", "a") as log_file:
                log_writer = csv.writer(log_file)
                if i == 0:  # 在第一个 epoch 添加表头
                    log_writer.writerow(["ID", "PSNR", "SSIM"])
                log_writer.writerow([i+1, psnr_value, ssim_value])
        
            genimg = (genimg * 255).astype(np.uint8)
            genimg_pil = Image.fromarray(genimg)
            genimg_pil.save(self.results_folder + f"imgs_gen/gen_epoch={self.epoch}_{i+1}.jpg", "JPEG", quality=100)


            cost_time = time.time()-start_time
            current, peak = tracemalloc.get_traced_memory()
            print(f"Current memory usage: {current / 1024**2} MB")
            print(f"Peak memory usage: {peak / 1024**2} MB")
            print(i+1,'次:',cost_time,'s')
            
        tracemalloc.stop()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", 
                        type=str, 
                        default="saved_models/generator_2025-01-17_499.pth", 
                        help="path of the model checkpoints")######
    parser.add_argument("--epoch", 
                        type=int, 
                        default=499, 
                        help="epoch of model checkpoints")
    parser.add_argument("--testimgs_folder", 
                        type=str, 
                        default="testimgs/", 
                        help="path of the low quality imgs")######
    parser.add_argument("--results_folder", 
                        type=str, 
                        default="results/", 
                        help="path of the gt images")######
    args = parser.parse_args()

    ReconImg = Recon(args)
    ReconImg.run()
