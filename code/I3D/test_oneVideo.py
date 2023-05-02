# -*- coding:utf-8 -*-
# @FileName :test_oneVideo.py
# @Time :2023/4/27 15:16
# @Author :Xiaofeng
import json
import random

import cv2
import numpy as np
import torch
from torchvision.transforms import transforms

import videotransforms
from datasets.nslt_dataset_all import video_to_tensor
from pytorch_i3d import InceptionI3d
from torch import nn


def load_video_to_rgb_frame(video_path):
    """
    这个函数将视频加载成一个由RGB帧组成的张量。函数会读取视频文件的每一帧
    """
    num_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for offset in range(num_frames):
        success, img = cv2.VideoCapture(video_path).read()
        w, h, c = img.shape
        # 将帧的大小调整为最小边长至少为226像素
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        # 将像素值归一化为[-1, 1]的范围，
        img = (img / 255.) * 2 - 1
        # 并将所有帧组合成一个形状为(num_frames, height, width, 3)的张量
        # 其中，num_frames表示视频中的帧数，height和width表示每一帧的高度和宽度
        frames.append(img)
    # if len(frames) > 128:
    #     mask = random.sample(range(len(frames)), 128)
    #     mask.sort()
    #     frames = [frames[i] for i in mask]

    return np.asarray(frames, dtype=np.float32)


def nslt(video_path):
    img = load_video_to_rgb_frame(video_path)
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    img = test_transforms(img)
    return video_to_tensor(img)


def get_action_class(split_file):
    classes = set()

    content = json.load(open(split_file))

    for vid in content.keys():
        class_id = content[vid]['action'][0]
        classes.add(class_id)
    return classes


if __name__ == '__main__':
    # 加载训练好的 PyTorch 模型
    # weights = 'archived/asl100/FINAL_nslt_100_iters=896_top1=65.89_top5=84.11_top10=89.92.pt'
    weights = 'checkpoints/nslt_100_001683_0.670000.pt'
    # 使用 OpenCV 中的 VideoCapture 函数加载视频，并使用您训练模型时使用的相同预处理步骤将视频帧转换为模型期望的格式
    video_path = "D:/Code/PythonCode/SignLanguageProject/WLASL/data/videos/28202.mp4"
    img_frame = nslt(video_path)

    # 对视频帧进行预测,使用 PyTorch 模型对处理后的视频帧进行预测，并将结果存储在一个数组中
    i3d = InceptionI3d(400, in_channels=3)
    i3d.load_state_dict(torch.load('weights/rgb_imagenet.pt'))
    i3d.replace_logits(100)
    i3d.load_state_dict(torch.load(weights))
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    i3d.eval()
    # t = img_frame.size(2) 帧数
    with torch.no_grad():
        img_frame = img_frame.unsqueeze(0)
        per_frame_logits = i3d(img_frame)

    # 将预测结果转换为手语动作标签
    # classes = get_action_class('preprocess/nslt_100.json')
    predictions = torch.mean(per_frame_logits, dim=2)[0]
    out_labels = np.argsort(predictions.cpu().detach().numpy())

    # 打印预测结果
    print('Predicted action labels top-1: ', torch.argmax(predictions).item())
    print('Predicted action labels top-5: ', out_labels[-5:])
    print('Predicted action labels top-10: ', out_labels[-10:])
