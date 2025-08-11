import os

# 获取当前目录
target_directory = '/home/dataset/DNS-Challenge/DNS-Challenge/datasets/noise'

# 清点文件数
file_count = len([file for file in os.listdir(target_directory) if os.path.isfile(os.path.join(target_directory, file))])

# 输出结果
print(f"目录 '{target_directory}' 中的文件数量为：{file_count}")
