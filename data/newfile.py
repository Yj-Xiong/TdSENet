# 定义要替换的路径和新路径
old_path = '../'
new_path = '/home/dataset/THUYG-20/'

# 读取文件
with open('file.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 替换路径并写入新文件
with open('dev.txt', 'w', encoding='utf-8') as new_file:
    for line in lines:
        new_line = line.replace(old_path, new_path)
        new_file.write(new_line)
