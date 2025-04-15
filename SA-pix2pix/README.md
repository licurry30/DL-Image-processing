# 开始项目
1. 准备数据集

      将下载好的数据，包括包含待复原的图像和GT图像文件夹放在*data*目录下
      
      准备好org.csv文件
  
2. 配置环境 

      conda create -n python3.8 python==3.8

      conda activate python3.8

      pip install -r requirements.txt

3. 训练

      python train.py
   
5. 测试
  
      python test.py
