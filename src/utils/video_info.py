import cv2,sys

if __name__ == "__main__":
    file_path = '/home/acsguser/Codes/SwinBERT/datasets/Violence/data/test/v=-etV57xZ4_I__#1_label_B4-0-0.mp4'
    cap = cv2.VideoCapture(file_path)
    if cap.isOpened():  # 当成功打开视频时cap.isOpened()返回True,否则返回False
        # get方法参数按顺序对应下表（从0开始编号)
        rate = cap.get(5)   # 帧速率
        FrameNumber = cap.get(7)  # 视频文件的帧数
        duration = FrameNumber/rate/60  # 帧速率/视频总帧数 是时间，除以60之后单位是分钟
        print("FPS:", rate)
        print("FrameNo:", FrameNumber)
        print("duration", duration)

        # 读取一帧进行保存
        # ret, frame = cap.read()
        # cv2.imwrite('/home/acsguser/Downloads/01_001.jpg', frame)
        # sys.exit(0)
