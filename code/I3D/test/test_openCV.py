# -*- coding = utf-8 -*-
# @Time : 2023/3/17 10:05
# @Author : xiaofeng
# @Description:
import math

import cv2
import numpy as np


def test_video_capture():
    video_path = '../../../data/videos/20157.mp4'
    vidcap = cv2.VideoCapture(video_path)
    frames = []
    start = 0
    # 读取视频中的总帧数
    total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    # 获取视频帧的宽
    w = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # 获取视频帧的高
    h = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # 获取视频帧的帧率
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    # 跳到某一感兴趣帧并从此帧开始读取,如从第360帧开始读取 vidcap.set(cv2.CAP_PROP_POS_FRAMES, 360)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for offset in range(min(60, int(total_frames - start))):
        # 获取帧画面
        success, img = vidcap.read()

        w, h, c = img.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

        if w > 256 or h > 256:
            img = cv2.resize(img, (math.ceil(w * (256 / w)), math.ceil(h * (256 / h))))

        img = (img / 255.) * 2 - 1

        frames.append(img)

    return np.asarray(frames, dtype=np.float32)


def stream_processing():
    # 获取VideoCapture类实例，读取视频文件
    fcap = cv2.VideoCapture('../../../data/videos/20157.mp4')

    # 设置摄像头分辨率的高
    fcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # 设置摄像头分辨率的宽
    fcap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # 跳到某一感兴趣帧并从此帧开始读取,如从第360帧开始读取
    fcap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 获取视频帧的宽
    w = fcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # 获取视频帧的高
    h = fcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # 获取视频帧的帧率
    fps = fcap.get(cv2.CAP_PROP_FPS)
    # 获取视频流的总帧数
    fcount = fcap.get(cv2.CAP_PROP_FRAME_COUNT)

    # 获取VideoWriter类实例
    writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), int(fps), (int(w), int(h)))

    # 判断是否正确获取VideoCapture类实例
    while fcap.isOpened():
        # 获取帧画面
        success, frame = fcap.read()
        while success:
            cv2.imshow("demo", frame)  ## 显示画面
            # 获取帧画面
            success, frame = fcap.read()

            # 保存帧数据
            writer.write(frame)

            if (cv2.waitKey(20) & 0xff) == ord('q'):  ## 等待20ms并判断是按“q”退出，相当于帧率是50hz，注意waitKey只能传入整数，
                break
        # 释放VideoCapture资源
        fcap.release()
    # 释放VideoWriter资源
    writer.release()
    cv2.destroyAllWindows()  ## 销毁所有opencv显示窗口


if __name__ == '__main__':
    stream_processing()
    imgs = test_video_capture()
