from PIL import Image
import numpy as np


def test2():
    # 创建一个NumPy数组，表示图像的像素值
    nparray = np.array([[ 99,  89,  95],
        [ 96,  86,  92],
        [102,  92,  98]])
    # 将NumPy数组转换为PIL图像
    img = Image.fromarray(nparray.astype(np.uint8))
    # 显示图像
    img.show()

if __name__ == '__main__':
    test2()
