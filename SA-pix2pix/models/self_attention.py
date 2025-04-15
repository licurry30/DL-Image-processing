import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttentionModule, self).__init__()
        self.in_channels = in_channels

        # Query, Key, and Value layers 卷积参数
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.output_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Compute Query, Key, and Value
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1) # 1 8 1024 --> 1 1024 8 -1代表自动推导该位置维度大小
        key = self.key_conv(x).view(batch_size, -1, height * width)# 1 8 1024
        value = self.value_conv(x).view(batch_size, -1, height * width) # 1 64 1024

        # Calculate attention weights
        #print(query.size(),key.size())
        attention_scores = torch.matmul(query, key)# √
        attention_scores = F.softmax(attention_scores, dim=-1)

        # Apply attention weights to the value
        #print(value.size(), attention_scores.size())
        attention_output = torch.matmul(value, attention_scores)
        attention_output = attention_output.view(batch_size, channels, height, width)
        output = self.output_conv(attention_output)

        return output

# 测试自注意力模块
if __name__ == "__main__":
    in_channels = 512
    input_data = torch.randn(1, in_channels, 256, 256)
    self_attention = SelfAttentionModule(in_channels)
    output = self_attention(input_data)
    print("Input shape:", input_data.shape)
    print("Output shape:", output.shape)



        
