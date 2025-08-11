import os
import shutil

# 定义输入文件和对应的新目录
file_mapping = {
    '/home/xyj/datasets/english/train_cleanfiles.txt': '/home/xyj/datasets/english/train/',
    # 'uyghur_train.txt': '/home/xyj/datasets/uyghur/train/',
    # 'uyghur_dev.txt': '/home/xyj/datasets/uyghur/dev/',
    # 'uyghur_test.txt': '/home/xyj/datasets/uyghur/test/'
}

# 遍历每个输入文件和对应的新目录
for input_file, new_directory in file_mapping.items():
    # 确保新目录存在
    os.makedirs(new_directory, exist_ok=True)

    # 读取文件列表
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 复制文件到新位置
    for line in lines:
        line = line.strip()  # 去掉行末的换行符和空格
        if line:  # 确保行不为空
            # 复制文件
            shutil.copy(line, new_directory)

print("文件复制完成，已输出到新的目录。")
