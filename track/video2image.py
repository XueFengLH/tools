# -*- coding: utf-8 -*-
# 使用本程序的方法：
# 1. 安装python3
# 2. 安装numpy， 命令 pip3 install numpy
# 3. 安装opencv-python， 命令：pip3 install opencv-python
# 4. 将该脚本放置在存储视频的文件夹中，利用python执行此脚本。

import os
import sys
import tqdm
# filedir = os.path.dirname(sys.argv[0])  # 获取脚本所在目录
# os.chdir(filedir)  # 将脚本所在的目录设置为工作目录
# wdir = os.getcwd()
# print('当前工作目录：{}\n'.format(wdir))  # 打印当前工作目录

import numpy as np
import cv2 as cv;
import re
from math import ceil



def video_to_image(video_path,save_path):
    print('\n-----------------------------------------------------------------------------')
    print('正在处理视频：{}\n'.format(video_path))

    video_dir = os.path.dirname(video_path)
    video_name = os.path.basename(video_path)

    image_dir = re.sub('\.', '_', video_name)
    image_dir = os.path.join(save_path, image_dir.replace('mp4','') + "/img")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    cap = cv.VideoCapture(video_path)
    # Define the codec and create VideoWriter object
    fps = cap.get(5)  # 视频帧率
    width = cap.get(3)
    height = cap.get(4)
    zhen_shu = cap.get(7)
    zhen_shu = int(zhen_shu)
    sample_rate = 1 # 采样率，目前设置为每秒取1帧。

    # print("视频  {}  的帧率为:{:.2f}  分辨率为:{} ✖ {}".format(video_path, fps, width, height))
    print("视频  {} 的帧率为：{}  分辨率为：{} * {}".format(video_path, fps, width, height))

    n = 0
    new_img_num = 0
    old_img_num = 0
    with tqdm.tqdm(total=zhen_shu-1) as pbar:
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                # cv.imshow('frame1', frame)
                n = n + 1
                pbar.update(1)
                if n % sample_rate == 0:

                    image_name = video_name + '_' + str(n) + '.jpg'
                    image_path = os.path.join(image_dir, image_name)
                    if os.path.exists(image_path):
                        print('{}  已经存在。'.format(image_path))
                        old_img_num += 1
                    else:
                        rotated_frame = cv.rotate(frame, cv.ROTATE_180)
                        # print(image_path)
                        cv.imwrite(image_path.replace('.mp4',''), rotated_frame)
                        new_img_num += 1

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        # Release everything if job is finished

        cap.release()
        print('视频处理完毕，共{}帧。每{}帧采1张，可生成{}张图片。\n'.format(n, sample_rate, ceil(n / sample_rate)))
        print('此前已有图片{}张，本次生成{}张。\n'.format(old_img_num, new_img_num))


ext = ('.h264', '.MOV')
# 遍历工作目录，并对其中指定格式的视频文件进行处理。
# def search(path):
# for p in os.listdir(path):
# p = os.path.join(path, p)
# if os.path.isdir(p):
# search(p)
# elif p.endswith(ext):
# video_to_image(p)

# search(wdir)
wdir = '/home/spring/nfs_client/face_data/video'
for dirpath, dirname, filename in os.walk(wdir):
    for f in filename:
        if f.endswith('.mp4'):
            video_path = os.path.join(dirpath, f)
            video_to_image(video_path,save_path='/home/spring/nfs_client/face_data/video_image')
