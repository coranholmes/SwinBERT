import cv2

if __name__ == "__main__":
    file_path = '/home/acsguser/Codes/SwinBERT/datasets/Crime/data/Testing_Normal_Videos_Anomaly/Normal_Videos_923_x264.mp4'
    cap = cv2.VideoCapture(file_path)
    if cap.isOpened():  # 当成功打开视频时cap.isOpened()返回True,否则返回False
        # get方法参数按顺序对应下表（从0开始编号)
        rate = cap.get(5)   # 帧速率
        FrameNumber = cap.get(7)  # 视频文件的帧数
        duration = FrameNumber/rate/60  # 帧速率/视频总帧数 是时间，除以60之后单位是分钟
        print("FPS:", rate)
        print("FrameNo:", FrameNumber)
        print("duration", duration)
