import glob
import os
import shutil


def combine_tfevents(log_dir, output_file):
    # 获取所有 tfevents 文件的完整路径
    tfevents_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))

    # 按照文件的修改时间排序
    tfevents_files.sort(key=os.path.getmtime)
 # 创建输出文件
    with open(output_file, 'wb') as outfile:
        for fname in tfevents_files:
            print(f"Processing file: {fname}")
            with open(fname, 'rb') as infile:
                shutil.copyfileobj(infile, outfile)  # 将内容写入到输出文件中

    print(f"Combined tfevents files into: {output_file}")

# 使用示例
log_directory = '/home/xyj/Experience/CMG-v1/src/SSL_Logs/Logs/MdGAN_normed'
output_filepath = '/home/xyj/Experience/CMG-v1/src/SSL_Logs/Logs/MdGANnormed/combined_events.tfevents'  # 输出文件路径
combine_tfevents(log_directory, output_filepath)