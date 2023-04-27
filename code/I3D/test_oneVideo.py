# -*- coding:utf-8 -*-
# @FileName :test_oneVideo.py
# @Time :2023/4/27 15:16
# @Author :Xiaofeng
import torch

# 加载训练好的 PyTorch 模型
model = torch.load('archived/asl100/FINAL_nslt_100_iters=896_top1=65.89_top5=84.11_top10=89.92.pt')
# 加载视频并将其转换为模型期望的格式
video_path = "D:/Code/PythonCode/SignLanguageProject/WLASL/data/videos/01226"
