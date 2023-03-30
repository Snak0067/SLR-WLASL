# -*- coding = utf-8 -*-
# @Time : 2023/3/14 13:42
# @Author : xiaofeng
# @Description:
import torch


def test_pytorch_gpu_version():
    print(torch.__version__)  # 注意是双下划线
    print(torch.cuda.is_available())


if __name__ == '__main__':
    test_pytorch_gpu_version()
