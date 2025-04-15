import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.nn as nn
import torch
from torchsummary import summary

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 4, 2, 1), 
                     nn.LeakyReLU(0.2, inplace=True), 
                     nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(1, 64, bn=False), #256
            *discriminator_block(64, 128), #128
            *discriminator_block(128, 256), #64
            *discriminator_block(256, 512), #32
            *discriminator_block(512, 512), #16
            nn.ZeroPad2d((1, 0, 1, 0)), #17*17
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, bias=False), # 16*16
        )

    def forward(self, x):
        output = self.model(x)
        return output
    



from thop import profile

if __name__ == '__main__':
    #/*****/模型可视化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    D = Discriminator().to(device)

    print('Discriminator:')
    summary(D,input_size=(1, 512, 512))


    
