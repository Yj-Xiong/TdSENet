import os
import random

import numpy as np
import torch
import torchaudio
from natsort import natsorted

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def calculate_power(waveform):
    return np.mean(waveform ** 2)


def add_noise_with_snr(clean_waveform, noise_waveform, snr_db):
    clean_power = calculate_power(clean_waveform)
    noise_power = calculate_power(noise_waveform)

    desired_noise_power = clean_power / (10 ** (snr_db / 10) + 1e-10)
    scaling_factor = np.sqrt(desired_noise_power / noise_power)
    adjusted_noise = noise_waveform * scaling_factor

    noisy_waveform = clean_waveform + adjusted_noise
    return noisy_waveform


def generate_noisy_dataset(clean_dir, noise_dir, output_dir, snr_range, num_samples, segment_length):
    clean_files = natsorted([os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith('.wav')])
    noise_files = natsorted([os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith('.wav')])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    generated_count = 0
    clean_file_number = 0  # 在外层定义
    used_clean_files = []
    count = 0
    # 创建一个目录专门保存干净语音数据
    clean_output_dir = os.path.join(output_dir, 'used_clean')
    if not os.path.exists(clean_output_dir):
        os.makedirs(clean_output_dir)

    for i in range(num_samples):

        for clean_file in clean_files:
            clean_waveform, sample_rate = torchaudio.load(clean_file)

            # Split clean waveform into segments of 10 seconds
            num_segments = clean_waveform.size(1) // (sample_rate * segment_length)
            clean_file_number += 1  # 在这里增加计数
            for i in range(num_segments):
                segment_start = i * sample_rate * segment_length
                segment_end = segment_start + sample_rate * segment_length
                clean_segment = clean_waveform[:, segment_start:segment_end]
                if clean_segment.size(1) < sample_rate * segment_length:
                    break

                # 从噪声文件列表中随机选择一个噪声文件
                noise_file = random.choice(noise_files)

                noise_waveform, _ = torchaudio.load(noise_file)

                # Ensure noise waveform is at least as long as clean segment
                if noise_waveform.size(1) < clean_segment.size(1):
                    noise_waveform = torch.cat([noise_waveform] * (clean_segment.size(1) // noise_waveform.size(1) + 1),
                                               dim=1)

                noise_waveform = noise_waveform[:, :clean_segment.size(1)]

                snr_db = random.randint(snr_range[0], snr_range[-1])  # Random SNR from -5dB to 15dB
                noisy_waveform = add_noise_with_snr(clean_segment.numpy()[0], noise_waveform.numpy()[0], snr_db)
                noisy_waveform = torch.Tensor(noisy_waveform).unsqueeze(0)

                # 修改输出文件名格式，将条件放在文件名前部
                output_file = os.path.join(output_dir, f"snr{snr_db}_{os.path.splitext(os.path.basename(clean_file))[0]}_seg{i}.wav")
                torchaudio.save(output_file, noisy_waveform, sample_rate)
                print(f"Generated noisy file: {output_file} with SNR: {snr_db} dB")

                # 保存使用的干净音频片段
                clean_output_file = os.path.join(clean_output_dir, f"{os.path.splitext(os.path.basename(clean_file))[0]}_seg{i}.wav")
                torchaudio.save(clean_output_file, clean_segment, sample_rate)

                generated_count += 1

            used_clean_files.append(clean_file)

        # Write used clean file paths to a text file
        with open(os.path.join(output_dir, 'used_clean_files.txt'), 'w') as f:
            f.write("\n".join(used_clean_files))
        count+=1
        if count == num_samples:
            break

    print(f"Generated {generated_count} noisy files in total, initially set {count} number of samples, {count-generated_count} files are not generated.")
    print(f"Count of {clean_file_number} clean files in total.")

# 指定路径和参数
snr_range = list(range(-5, 16))  # SNR从-5到15
# print(f"SNR range: {snr_range}")
clean_dir = '/home/dataset/DNS-Challenge/DNS-Challenge/datasets/clean'
noise_dir = '/home/dataset/DNS-Challenge/DNS-Challenge/datasets/noise'
output_dir = os.path.join('./DNS_noisy', f"SNR=[{snr_range[0]},{snr_range[-1]}]")
segment_length = 10  # 10 seconds

# 计算生成样本数量 (3000小时 * 3600秒 //小时 / 10秒每片段)
num_samples = (3000 * 3600)/ segment_length
print(f"Number of samples to be generated: {num_samples}")
generate_noisy_dataset(clean_dir, noise_dir, output_dir, snr_range, int(num_samples), segment_length)
