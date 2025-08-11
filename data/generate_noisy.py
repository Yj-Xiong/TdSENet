import shutil

import torch

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# 计算功率的方法
def calculate_power(waveform):
    return np.mean(waveform ** 2)

# 添加噪声到干净的波形
def add_noise_with_snr(clean_waveform, noise_waveform, snr_db):
    clean_power = calculate_power(clean_waveform)
    noise_power = calculate_power(noise_waveform)

    desired_noise_power = clean_power / (10 ** (snr_db / 10) + 1e-10)
    scaling_factor = np.sqrt(desired_noise_power / noise_power)
    adjusted_noise = noise_waveform * scaling_factor

    noisy_waveform = clean_waveform + adjusted_noise
    return noisy_waveform

# 生成带噪声的数据集
# def generate_noisy_dataset(clean_dir, noise_dir, output_dir, snr_range, num_samples):
#     clean_files = natsorted([os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith('.wav')])
#     noise_files = natsorted([os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith('.wav')])
#     used_noise_files = []
#
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     generated_count = 0
#     used_clean_files = []
#
#     selected_clean_files = random.sample(clean_files, num_samples)
#
#     for clean_file in selected_clean_files:
#         clean_waveform, sample_rate = torchaudio.load(clean_file)
#
#         available_noise_files = [f for f in noise_files if f not in used_noise_files]
#         noise_file = random.choice(available_noise_files)
#         used_noise_files.append(noise_file)
#
#         noise_waveform, _ = torchaudio.load(noise_file)
#
#         if noise_waveform.size(1) < clean_waveform.size(1):
#             noise_waveform = torch.cat([noise_waveform] * (clean_waveform.size(1) // noise_waveform.size(1) + 1), dim=1)
#
#         noise_waveform = noise_waveform[:, :clean_waveform.size(1)]
#
#         snr_db = random.choice(snr_range)
#         noisy_waveform = add_noise_with_snr(clean_waveform.numpy()[0], noise_waveform.numpy()[0], snr_db)
#         noisy_waveform = torch.Tensor(noisy_waveform).unsqueeze(0)
#
#         output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(clean_file))[0]}_snr{snr_db}.wav")
#         torchaudio.save(output_file, noisy_waveform, sample_rate)
#         print(f"Generated noisy file: {output_file} with SNR: {snr_db} dB")
#
#         generated_count += 1
#         used_clean_files.append(clean_file)
#
#     with open(os.path.join(output_dir, 'used_clean_files.txt'), 'w') as f:
#         f.write("\n".join(used_clean_files))
#
#     print(f"Generated {generated_count} noisy files in total.")
import os
import numpy as np
import torch
import torchaudio
import random
from natsort import natsorted

def generate_noisy_dataset(clean_dir, noise_dir, output_dir, snr_range):
    # 获取并排序所有干净和噪声文件
    clean_files = natsorted([os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith('.wav')])
    noise_files = natsorted([os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith('.wav')])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    generated_count = 0

    # 使用每个干净文件
    for clean_file in clean_files:
        clean_waveform, sample_rate = torchaudio.load(clean_file)

        # 每次随机选择噪声文件
        noise_file = random.choice(noise_files)
        noise_waveform, _ = torchaudio.load(noise_file)

        if noise_waveform.size(1) < clean_waveform.size(1):
            noise_waveform = torch.cat([noise_waveform] * (clean_waveform.size(1) // noise_waveform.size(1) + 1), dim=1)

        noise_waveform = noise_waveform[:, :clean_waveform.size(1)]

        snr_db = random.choice(snr_range)
        noisy_waveform = add_noise_with_snr(clean_waveform.numpy()[0], noise_waveform.numpy()[0], snr_db)
        noisy_waveform = torch.Tensor(noisy_waveform).unsqueeze(0)

        # 根据原始文件名生成输出文件名，确保文件名一致
        base_filename = os.path.splitext(os.path.basename(clean_file))[0]
        output_file = os.path.join(output_dir, f"{base_filename}.wav")  # 保持文件名一致，不带 SNR 信息

        torchaudio.save(output_file, noisy_waveform, sample_rate)
        print(f"Generated noisy file: {output_file} with SNR: {snr_db} dB")

        generated_count += 1

    print(f"Generated {generated_count} noisy files in total.")



# def add_noise_with_snr(clean_waveform, noise_waveform, snr_db):
#     # 添加噪声的逻辑
#     # 请根据您的需求实现这个函数
#     pass

def generate_noisy_data_db(clean_dir, noise_dir, output_base_dir, snr_range):
    # 获取并排序所有干净和噪声文件
    clean_files = natsorted([os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith('.wav')])
    noise_files = natsorted([os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith('.wav')])
    num_clean_files = len(clean_files)
    # 使用每个 SNR 值生成带噪声文件
    for snr_db in snr_range:

        output_dir = os.path.join(output_base_dir, f"SNR_{snr_db}dB")
        num_clean_files += len(clean_files)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for clean_file in clean_files:
            clean_waveform, sample_rate = torchaudio.load(clean_file)

            # 随机选择噪声文件
            noise_file = random.choice(noise_files)
            noise_waveform, _ = torchaudio.load(noise_file)

            if noise_waveform.size(1) < clean_waveform.size(1):
                noise_waveform = torch.cat([noise_waveform] * (clean_waveform.size(1) // noise_waveform.size(1) + 1), dim=1)

            noise_waveform = noise_waveform[:, :clean_waveform.size(1)]

            noisy_waveform = add_noise_with_snr(clean_waveform.numpy()[0], noise_waveform.numpy()[0], snr_db)
            noisy_waveform = torch.Tensor(noisy_waveform).unsqueeze(0)

            # 根据原始文件名生成输出文件名，确保文件名一致
            base_filename = os.path.splitext(os.path.basename(clean_file))[0]
            output_file = os.path.join(output_dir, f"{base_filename}.wav")  # 保持文件名一致，不带 SNR 信息

            torchaudio.save(output_file, noisy_waveform, sample_rate)
            print(f"Generated noisy file: {output_file} with SNR: {snr_db} dB")

    print(f"Generated {num_clean_files} noisy files with SNR range: {snr_range} dB in total.")



# 从噪声目录中随机选择 200 条噪声并复制
def select_and_copy_noise_files(noise_dir, output_noise_dir, num_noise_samples=200):
    noise_files = natsorted([os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith('.wav')])

    if not os.path.exists(output_noise_dir):
        os.makedirs(output_noise_dir)

    selected_noise_files = random.sample(noise_files, min(num_noise_samples, len(noise_files)))

    for noise_file in selected_noise_files:
        shutil.copy(noise_file, output_noise_dir)
        print(f"Copied noise file: {noise_file} to {output_noise_dir}")

# # 创建一个单独的输出目录来存放选中的噪声文件
# output_noise_dir = '/home/xyj/Experience/DNS-Challenge/random400_noise'
# select_and_copy_noise_files(noise_dir, output_noise_dir, num_noise_samples=400)
#
# # 从噪声目录中选择并复制噪声文件
# select_and_copy_noise_files(noise_dir, output_noise_dir)
# 指定路径和SNR范围
# snr_range = [-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10]
snr_range = [-10,-7.5,-5,-2.5]
clean_dir = '/home/dataset/Voicebank/noisy-vctk-16k/clean_testset_wav_16k/'
# noise_dir = '/home/dataset/DNS-Challenge/DNS-Challenge/datasets/noise/'
# noise_dir = '/home/xyj/Experience/DNS-Challenge/random400_noise'
noise_dir = '/home/xyj/NoiseX-92/'
output_dir = os.path.join('/home/xyj/datasets/english/vb_babble',)  #f"SNR:{snr_range}"

# 生成带噪声的数据集
# generate_noisy_dataset(clean_dir, noise_dir, output_dir, snr_range)
generate_noisy_data_db(clean_dir, noise_dir, output_dir, snr_range)
# class CustomDataset(Dataset):
#     def __init__(self, clean_files_path, transform=None):
#         with open(clean_files_path, 'r') as f:
#             self.clean_files = f.read().splitlines()
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.clean_files)
#
#     def __getitem__(self, idx):
#         clean_file = self.clean_files[idx]
#         clean_waveform, sample_rate = torchaudio.load(clean_file)
#
#         if self.transform:
#             clean_waveform = self.transform(clean_waveform)
#
#         return clean_waveform, sample_rate
