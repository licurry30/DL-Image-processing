# 部署yolov8

**配置环境**

**cuda匹配的torch就可**

**这里：**

**cuda==11.4**

**python==3.8** #尽量用3.8

**torch==1.10**

**然后**

**git clone 项目**

**进入main目录**

**pip install e.** 安装再本地

**yolov8n.pt yolov11n.pt yolov8s.pt yolov11s.pt 要下载**

**训练和推理脚本：****

**例如：**

**yolo train data=my_data/data.yaml model=yolov8n.pt epochs=100 batch=8 workers=0 device=0**

**yolo predict model=runs/detect/train/weights/best.pt source='Quicker_20220930_180938.png' device=0**

**车牌识别项目：** https://github.com/SYQ50001/plate-recognition_YOLOv8_CNN

**将训练好的权重放到weights/**

**运行**detect_rec_plate.py**即可** 注意：修改路径
