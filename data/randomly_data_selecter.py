import json
import os

# 指定音频文件所在目录
audio_directory = '/home/dataset/THCHS-30/data_thchs30/dev/'
output_directory = '/home/xyj/datasets/chinese/'
output_file = 'Cfiles.json'

# 确保输出目录存在
os.makedirs(output_directory, exist_ok=True)


# 获取目录中所有音频文件
all_audio_files = [f for f in os.listdir(audio_directory) if f.endswith('.wav')]

# 去除文件后缀并形成列表
audio_file_names = [os.path.splitext(f)[0] for f in all_audio_files]

# 如果文件存在，则以追加模式打开文件
if os.path.exists(os.path.join(output_directory, output_file)):
    with open(os.path.join(output_directory, output_file), 'r+', encoding='utf-8') as file:
        # 读取现有内容
        existing_data = json.load(file)
        # 合并新内容
        existing_data.extend(audio_file_names)
        # 移动指针到文件开头
        file.seek(0)
        # 写回更新后的数据
        json.dump(existing_data, file, ensure_ascii=False, indent=4)
        file.truncate()  # 截断文件，移除旧数据
else:
    # 文件不存在，创建新文件
    with open(os.path.join(output_directory, output_file), 'w', encoding='utf-8') as file:
        json.dump(audio_file_names, file, ensure_ascii=False, indent=4)

print(f"生成的文件列表已保存到 {os.path.join(output_directory, output_file)}。")
# # 获取目录中所有音频文件
# all_audio_files = [f for f in os.listdir(audio_directory) if f.endswith('.wav')]

# 随机选择 400 条音频文件
# selected_audio_files = random.sample(all_audio_files, min(400, len(all_audio_files)))

# 保存到文本文件
# with open(os.path.join(output_directory, output_file), 'w', encoding='utf-8') as file:
#     for audio_file in selected_audio_files:
#         file.write(os.path.join(audio_directory, audio_file) + '\n')

# print(f"生成的文件列表已保存到 {os.path.join(output_directory, output_file)}。")
