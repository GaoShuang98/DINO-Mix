import os
from PIL import Image

def crop_images(folder_path):
    # 获取文件夹中所有 JPG 图像的文件名
    image_files = [file for file in os.listdir(folder_path) if file.endswith('.jpg')]

    for file in image_files:
        # 构建图像的完整路径
        image_path = os.path.join(folder_path, file)

        # 打开图像文件
        image = Image.open(image_path)

        # 获取图像的宽度和高度
        width, height = image.size

        # 定义剪切区域
        left = 80
        top = 0
        right = width - 80
        bottom = height

        # 剪切图像
        cropped_image = image.crop((left, top, right, bottom))

        # 保存剪切后的图像
        cropped_image.save(image_path)

        # 关闭图像文件
        image.close()

if __name__ == '__main__':

    # 调用函数来剪切图像
    folder_path = r'C:\Users\Administrator\Desktop\hgs图像检索文件\overleaf_DINO_Mix_new_grad_pics'
    crop_images(folder_path)