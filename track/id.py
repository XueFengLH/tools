import cv2
import numpy as np
import os
# 读取图片

from PIL import Image, ImageDraw, ImageFont

import cv2# 读取图像
def id(img,id):
    image = cv2.imread(img)# 获取图像中心点
    height, width, _ = image.shape
    center_x = width // 2
    center_y = height // 2# 添加序号
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 50
    font_color = (255, 255, 255)
    thickness = 80
    cv2.putText(image, id, (center_x, center_y), font, font_scale, font_color, thickness, cv2.LINE_AA)# 显示或保存图像
    # cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(img, image)
if __name__ == '__main__':
    images_path = '/home/spring/图片/id'
    for img in os.listdir(images_path):
        id(os.path.join(images_path, img),img.replace('.jpg',""))
        # add_label(os.path.join(images_path, img), img.replace('.jpg', ''))
# image_path = 'path_to_your_image.jpg'
# output_path = 'path_to_save_image.jpg'
# filename = 'your_filename.jpg'