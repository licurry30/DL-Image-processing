import cv2
import torch
import streamlit as st
import numpy as np
from PIL import Image, ImageTk
from plate_recognition.plate_rec import init_model
from detect_rec_plate import load_model, det_rec_plate, draw_result
import tempfile
# Streamlit Web App
st.title("车牌识别系统")

# 左侧栏：模型选择
st.sidebar.header("模型选择")
detect_model_option = st.sidebar.selectbox("选择检测模型", ["yolov8s", "yolov8n"])  # 假设有多种检测模型
plate_rec_model_option = st.sidebar.selectbox("选择识别模型", ["plate_rec_color"])  # 假设有多种识别模型

# 摄像头选择
camera_id = st.sidebar.number_input("摄像头ID", min_value=0, max_value=10, value=0, step=1)

# 上传文件或视频选择
media_type = st.sidebar.radio("选择媒体类型", ("图片", "视频"))

# 动态加载模型
# @st.cache(allow_output_mutation=True)
def load_selected_models(detect_model_option, plate_rec_model_option):
    # 加载检测模型
    if detect_model_option == "yolov8s":
        detect_model = load_model("weights/yolov8s.pt", device)
    elif detect_model_option == "yolov8n":
        detect_model = load_model("weights/yolov8n.pt", device)
    else:
        raise ValueError("未知的检测模型选择")
    
    # 加载识别模型
    if plate_rec_model_option == "plate_rec_color":
        plate_rec_model = init_model(device, "weights/plate_rec_color.pth", is_color=True)
    else:
        raise ValueError("未知的识别模型选择")
    
    return detect_model, plate_rec_model

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
detect_model, plate_rec_model = load_selected_models(detect_model_option, plate_rec_model_option)

# 图片上传功能
if media_type == "图片":
    uploaded_file = st.sidebar.file_uploader("选择图片", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(('jpg', 'png', 'jpeg')):
                # 显示上传的图片
                file_bytes = uploaded_file.read()
                img_array = np.frombuffer(file_bytes, np.uint8)  # 将字节流转换为 numpy 数组
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # 解码为 OpenCV 图像格式

                image = Image.open(uploaded_file)
                image = image.resize((640, 480), Image.Resampling.LANCZOS)
                # image = ImageTk.PhotoImage(image)
                st.image(image, caption="上传的图片", use_container_width=True)
            else:
                st.error("文件格式错误，请上传 jpg, png 或 jpeg 格式的图片")
        except Exception as e:
            st.error(f"发生错误：{e}")
    else:
        st.info("请上传格式合适的车牌图片进行检测")

# 视频上传功能
if media_type == "视频":
    uploaded_video = st.sidebar.file_uploader("选择视频文件", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        # 显示视频上传
        st.video(uploaded_video)

    # 将上传的视频文件保存到临时文件
        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(uploaded_video.read())
        temp_video.close()
    else:
        st.info("请上传格式合适的车辆视频进行检测")

# 操作说明
st.sidebar.text("请选择图片并点击‘开始运行’按钮，进行车牌识别。")

# 开始识别按钮
if st.button('开始识别'):
    if media_type == "图片" and uploaded_file is not None:
        if 'img' in locals():  # 检查 img 是否已定义
            # 读取图像
            img_ori = img.copy()

            # 车牌检测与识别
            result_list = det_rec_plate(img, img_ori, detect_model, plate_rec_model)
            img, result_str = draw_result(img, result_list)

            # 显示处理后的图像和识别结果
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            # img = ImageTk.PhotoImage(img)
            st.image(img, caption="识别后的图像", use_container_width=True)
            st.success(f"识别结果：{result_str}")
        else:
            st.error("请先上传有效的图片文件以进行车牌识别")

    elif media_type == "视频" and uploaded_video is not None:
        # 处理视频文件，读取并显示车牌识别结果
        cap = cv2.VideoCapture(temp_video.name)
        if not cap.isOpened():
            st.error("无法打开视频文件")
        else:
            stframe = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # 车牌检测与识别
                result_list = det_rec_plate(frame, frame, detect_model, plate_rec_model)
                frame, result_str = draw_result(frame, result_list)

                # 显示处理后的帧
                stframe.image(frame, channels="BGR", use_container_width=True)
                st.success(f"识别结果：{result_str}")
            cap.release()
    else:
        st.error("请先选择有效的图片或视频进行识别")
