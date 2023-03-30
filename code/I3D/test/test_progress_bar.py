# -*- coding = utf-8 -*-
# @Time : 2023/3/13 16:32
# @Author : xiaofeng
# @Description: 测试进度条
from time import sleep
from tqdm import tqdm


def progress_bar():
    for i in tqdm(range(1, 500)):
        sleep(0.01)


if __name__ == '__main__':
    progress_bar()
