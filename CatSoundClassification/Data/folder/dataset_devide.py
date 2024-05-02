import os
import shutil
import random


def move_specific_files(source_folder, target_folder, keyword, fraction=0.1):
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历源文件夹中的每个类别文件夹
    for class_folder in os.listdir(source_folder):
        class_path = os.path.join(source_folder, class_folder)

        # 只处理文件夹
        if os.path.isdir(class_path):
            # 获取所有包含关键字的文件
            files = [f for f in os.listdir(class_path) if keyword in f]
            random.shuffle(files)  # 打乱文件列表

            # 计算要移动的文件数
            num_to_move = int(len(files) * fraction)
            moved_files = files[:num_to_move]

            # 创建目标文件夹
            target_class_folder = os.path.join(target_folder, class_folder)
            if not os.path.exists(target_class_folder):
                os.makedirs(target_class_folder)

            # 移动文件
            for file in moved_files:
                shutil.move(os.path.join(class_path, file), os.path.join(target_class_folder, file))

if __name__ == '__main__':
    # 使用函数
    source_folder = './Sound_T'
    target_folder = './SoundTest'  # 这里设置你的目标文件夹路径
    keyword = 'aug1(1)'  # 设置筛选的关键字
    move_specific_files(source_folder, target_folder, keyword)
