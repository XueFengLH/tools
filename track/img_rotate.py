import os
from PIL import Image
from tqdm import tqdm
# 定义输入和输出文件夹路径
input_folder = input("Input folder: ")
output_folder = input_folder

# 确保输出文件夹存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有文件
for filename in tqdm(os.listdir(input_folder)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # 检查文件扩展名
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # 打开图片
        img = Image.open(input_path)

        # 翻转图片180度
        rotated_img = img.rotate(180)

        # 保存翻转后的图片
        rotated_img.save(output_path)

print("所有图片翻转完成！")
