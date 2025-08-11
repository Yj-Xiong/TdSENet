import argparse
import os
import json,sys
import numpy as np
import soundfile as sf
import torch
import torchaudio
from natsort import natsorted
from models import generator_td as generator
# from models import Td_SENet as generator
from tools.gtcrn_compute_metrics import *
from utils import *
from tqdm import tqdm
from rich.progress import track

@torch.no_grad()
def enhance_one_track(model, audio_path, saved_dir, cut_len, n_fft=400, hop=100, save_tracks=False,device='cuda:0'):
    # 使用split方法分割字符串
    device_type, device_index = device.split(':')

    # 将device_index转换为整数
    device_number = int(device_index)
    name = os.path.split(audio_path)[-1]
    noisy, sr = torchaudio.load(audio_path)
    assert sr == 16000
    noisy = noisy.to(device)
#归一化
    c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
    noisy = torch.transpose(noisy, 0, 1)
    noisy = torch.transpose(noisy * c, 0, 1)

    length = noisy.size(-1)     #320
    # print(length)
    frame_num = int(np.ceil(length / 100))# 4
    padded_len = frame_num * 100    #400
    padding_len = padded_len - length#400-320=80
    noisy = torch.cat([noisy, noisy[:, :padding_len]], dim=-1)
    if padded_len > cut_len:
        batch_size = int(np.ceil(padded_len/cut_len))
        while 100 % batch_size != 0:
            batch_size += 1
        noisy = torch.reshape(noisy, (batch_size, -1))

    noisy_spec = torch.stft(noisy, n_fft, hop, window=torch.hamming_window(n_fft).to(device), onesided=True)
    noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
    est_real, est_imag, _ = model(noisy_spec)
    est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)

    est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
    est_audio = torch.istft(est_spec_uncompress, n_fft, hop, window=torch.hamming_window(n_fft).to(device),
                            onesided=True)
    est_audio = est_audio/c
    est_audio = torch.flatten(est_audio)[:length].cpu().numpy()
    # assert len(est_audio) == length
    if save_tracks:
        saved_path = os.path.join(saved_dir, name)
        sf.write(saved_path, est_audio, sr)
    torch.cuda.set_device(device_number)
    torch.cuda.empty_cache()
    return est_audio, length


def enhanced1(model_path, noisy_dir, save_tracks, saved_dir,device='cuda:0'):
    torch.device(device)
    map_location = torch.device(device)
    n_fft = 400
    model = generator.TSCNet(num_channel=64, num_features=n_fft//2+1).to(device)
    print('Loading model from {}'.format(model_path))
    model.load_state_dict((torch.load(model_path,map_location=map_location)))
    model.eval()
    print('Complete loading.')
    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)

    audio_list = os.listdir(noisy_dir)

    # 仅保留以 .wav 结尾的文件
    audio_list = [audio for audio in audio_list if audio.endswith('.wav')]
    audio_list = natsorted(audio_list)
    num = len(audio_list)
    for audio in tqdm(audio_list, desc="Enhancement Processing"):
        noisy_path = os.path.join(noisy_dir, audio)
        est_audio, length = enhance_one_track(model, noisy_path, saved_dir, 16000 *8, n_fft, n_fft // 4, save_tracks)


def evaluation( noisy_dir, clean_dir,  saved_dir):


    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)

    audio_list = os.listdir(noisy_dir)

    # 仅保留以 .wav 结尾的文件
    audio_list = [audio for audio in audio_list if audio.endswith('.wav')]
    audio_list = natsorted(audio_list)
    num = len(audio_list)
    metrics_total = np.zeros(6)
    for i, audio in enumerate(audio_list):
        est_path = os.path.join(saved_dir, audio)
        est_audio, sr = sf.read(est_path)
        clean_path = os.path.join(clean_dir, audio)
        clean_audio, sr = sf.read(clean_path)
        assert sr == 16000
        metrics = compute_metrics(clean_audio, est_audio, sr, 0)
        metrics = np.array(metrics)
        metrics_total += metrics

    metrics_avg = metrics_total / num
    print('pesq: ', metrics_avg[0], 'csig: ', metrics_avg[1], 'cbak: ', metrics_avg[2], 'covl: ',
          metrics_avg[3], 'ssnr: ', metrics_avg[4], 'stoi: ', metrics_avg[5])


def sisnr_evaluation(noisy_dir, clean_dir, saved_dir):
    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)

    audio_list = os.listdir(noisy_dir)
    audio_list = [audio for audio in audio_list if audio.endswith('.wav')]
    audio_list = natsorted(audio_list)

    metrics_total = np.zeros(8)
    num = len(audio_list)

    # 用于保存每个指标的单次结果，计算标准差
    metrics_list = []

    for audio in tqdm(audio_list, desc="Evaluation Processing"):
        est_path = os.path.join(saved_dir, audio)
        est_audio, sr = sf.read(est_path)

        clean_path = os.path.join(clean_dir, audio)
        clean_audio, sr = sf.read(clean_path)

        assert sr == 16000
        metrics = compute_all_metrics(clean_audio, est_audio, sr, 0)
        metrics = np.array(metrics)

        metrics_total += metrics
        metrics_list.append(metrics)  # 添加指标到列表

        # # 显示当前进度百分比
        # progress_percentage = (i + 1) / num * 100
        # sys.stdout.write(f'\rEvaluation Processing {i + 1}/{num} ({progress_percentage:.2f}%): {audio}')
        # sys.stdout.flush()

    print()

    metrics_avg = metrics_total / num
    metrics_std = np.std(metrics_list, axis=0)  # 计算标准差

    # 打印平均值和标准差
    print(f'pesq: {metrics_avg[0]:.4f} ± {metrics_std[0]:.4f}')
    print(f'csig: {metrics_avg[1]:.4f} ± {metrics_std[1]:.4f}')
    print(f'cbak: {metrics_avg[2]:.4f} ± {metrics_std[2]:.4f}')
    print(f'covl: {metrics_avg[3]:.4f} ± {metrics_std[3]:.4f}')
    print(f'ssnr: {metrics_avg[4]:.4f} ± {metrics_std[4]:.4f}')
    print(f'sisnr: {metrics_avg[5]:.4f} ± {metrics_std[5]:.4f}')
    print(f'sisdr: {metrics_avg[6]:.4f} ± {metrics_std[6]:.4f}')
    print(f'stoi: {metrics_avg[7]:.4f} ± {metrics_std[7]:.4f}')

    # 保存结果
    result_file_path = os.path.join(saved_dir, 'metrics_avg.json')
    results = {
        'pesq': float(metrics_avg[0]),
        'csig': float(metrics_avg[1]),
        'cbak': float(metrics_avg[2]),
        'covl': float(metrics_avg[3]),
        'ssnr': float(metrics_avg[4]),
        'sisnr': float(metrics_avg[5]),
        'sisdr': float(metrics_avg[6]),
        'stoi': float(metrics_avg[7]),
        'pesq_std': float(metrics_std[0]),
        'csig_std': float(metrics_std[1]),
        'cbak_std': float(metrics_std[2]),
        'covl_std': float(metrics_std[3]),
        'ssnr_std': float(metrics_std[4]),
        'sisnr_std': float(metrics_std[5]),
        'sisdr_std': float(metrics_std[6]),
        'stoi_std': float(metrics_std[7])
    }

    # 如果文件存在，加载已有数据并追加新结果
    if os.path.exists(result_file_path):
        with open(result_file_path, 'r') as json_file:
            all_results = json.load(json_file)
    else:
        all_results = []

    # 追加当前实验的结果
    all_results.append(results)

    # 保存更新后的结果
    with open(result_file_path, 'w') as json_file:
        json.dump(all_results, json_file, indent=4)

def noisy_evaluation(noisy_dir, clean_dir, saved_dir):
    # if not os.path.exists(saved_dir):
    #     os.mkdir(saved_dir)

    audio_list = os.listdir(noisy_dir)
    # 仅保留以 .wav 结尾的文件
    audio_list = [audio for audio in audio_list if audio.endswith('.wav')]
    audio_list = natsorted(audio_list)

    metrics_total = np.zeros(8)
    num = len(audio_list)
    # 用于保存每个指标的单次结果，计算标准差
    metrics_list = []
    for audio in track(audio_list):
        est_path = os.path.join(noisy_dir, audio)
        est_audio, sr = sf.read(est_path)

        clean_path = os.path.join(clean_dir, audio)
        clean_audio, sr = sf.read(clean_path)

        assert sr == 16000
        metrics = compute_all_metrics(clean_audio, est_audio, sr, 0)
        metrics = np.array(metrics)

        metrics_total += metrics
        metrics_list.append(metrics)  # 添加指标到列表

        # # 显示当前进度百分比
        # progress_percentage = (i + 1) / num * 100
        # sys.stdout.write(f'\rEvaluation Processing {i + 1}/{num} ({progress_percentage:.2f}%): {audio}')
        # sys.stdout.flush()

    # print()

    metrics_avg = metrics_total / num
    metrics_std = np.std(metrics_list, axis=0)  # 计算标准差

    # 打印平均值和标准差
    print(f'pesq: {metrics_avg[0]:.4f} ± {metrics_std[0]:.4f}')
    print(f'csig: {metrics_avg[1]:.4f} ± {metrics_std[1]:.4f}')
    print(f'cbak: {metrics_avg[2]:.4f} ± {metrics_std[2]:.4f}')
    print(f'covl: {metrics_avg[3]:.4f} ± {metrics_std[3]:.4f}')
    print(f'ssnr: {metrics_avg[4]:.4f} ± {metrics_std[4]:.4f}')
    print(f'sisnr: {metrics_avg[5]:.4f} ± {metrics_std[5]:.4f}')
    print(f'sisdr: {metrics_avg[6]:.4f} ± {metrics_std[6]:.4f}')
    print(f'stoi: {metrics_avg[7]:.4f} ± {metrics_std[7]:.4f}')

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default='/home/xyj/Experiment/CMG-v1/src/ckpt/multi_Td-epoch53-pesq:3.141-loss_sum:0.5601:-g:0.558-d:0.002',
                    help="the path where the model is saved")

# parser.add_argument("--test_dir", type=str, default='/home/dataset/Voicebank/noisy-vctk-16k/',
#                     help="noisy tracks dir to be enhanced")
# parser.add_argument("--model_path", type=str, default='/home/xyj/Experiment/CMG-v1/src/TdNet_VB0319/epoch16-pesq:3.543-loss_sum:0.5108:-g:0.506-d:0.004',
#                     help="the path where the model is saved")
#
parser.add_argument("--test_dir", type=str, default='/home/xyj/datasets/uyghur/',
                    help="noisy tracks dir to be enhanced")
parser.add_argument("--clean_dir", type=str, default='/home/xyj/datasets/uyghur/',
                    help="noisy tracks dir to be enhanced")
parser.add_argument("--save_tracks", type=str, default=True, help="save predicted tracks or not")
parser.add_argument("--save_dir", type=str, default='../Td_Multi-Enh_audio_U', help="where enhanced tracks to be saved")
args = parser.parse_args()


if __name__ == '__main__':
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    noisy_dir = os.path.join(args.test_dir, 'test_noisy')
    clean_dir = os.path.join(args.clean_dir, 'test')
    # noisy_dir = os.path.join(args.test_dir, 'noisy_testset_wav_16k')
    # clean_dir = os.path.join(args.test_dir, 'clean_testset_wav_16k')
    # noisy_dir = args.test_dir
    # noisy_dir = '/home/xyj/datasets/chinese/test_noisy'
    #         noisy_dir = os.path.join(args.test_dir, noisy_dir)
    # # #增强算法1处理
    # noisy_dir = '/home/xyj/datasets/chinese/dev_noisy'
    enhanced1(model_path = args.model_path, noisy_dir = noisy_dir,
               save_tracks = args.save_tracks, saved_dir = args.save_dir)

    # # #得到增强算法1的评价指标
    # sisnr_evaluation(noisy_dir=noisy_dir, clean_dir='/home/dataset/Voicebank/noisy-vctk-16k/clean_testset_wav_16k/',
    #                             saved_dir=args.save_dir)
    sisnr_evaluation( noisy_dir=noisy_dir, clean_dir = clean_dir,
               saved_dir=args.save_dir)
    # # 列出当前目录下的目录名称列表
    # noisy_dir_list = [d for d in os.listdir(args.test_dir) if os.path.isdir(os.path.join(args.test_dir, d))]
    #
    # for noisy_dir in noisy_dir_list:
    #     noisy_dir = os.path.join(args.test_dir, noisy_dir)
    #     output_dir = os.path.join(args.save_dir, os.path.split(noisy_dir)[-1])
    #
    #     if not os.path.exists(args.save_dir):
    #         os.mkdir(args.save_dir)
    #         os.mkdir(output_dir)
    # # #增强算法1处理
    #     enhanced1(model_path = args.model_path, noisy_dir = noisy_dir,
    #            save_tracks = args.save_tracks, saved_dir = output_dir)
    #
    # # #得到增强算法1的评价指标
    #     sisnr_evaluation( noisy_dir=noisy_dir, clean_dir = '/home/dataset/Voicebank/noisy-vctk-16k/clean_testset_wav_16k/',
    #            saved_dir=output_dir)
#
# # parser = argparse.ArgumentParser()
# # parser.add_argument("--model_path", type=str, default='/home/xyj/Experiment/epoch66-pesq:3.331-loss_sum:0.5180:-g:0.516-d:0.001',
# #                     help="the path where the model is saved")
# #
# parser.add_argument("--test_dir", type=str, default='/home/xyj/datasets/chinese/dev_noisy_db/',
#                     help="noisy tracks dir to be enhanced")
# parser.add_argument("--save_tracks", type=str, default=True, help="save predicted tracks or not")
# parser.add_argument("--save_dir", type=str, default='../Enh_audio-tdse_chineseDNS400', help="where enhanced tracks to be saved")
# args = parser.parse_args()
#
#
# if __name__ == '__main__':
#     if not os.path.exists(args.save_dir):
#         os.mkdir(args.save_dir)
#
#     # # 列出当前目录下的目录名称列表
#     noisy_dir_list = [d for d in os.listdir(args.test_dir) if os.path.isdir(os.path.join(args.test_dir, d))]
#
#     for noisy_dir in noisy_dir_list:
#         noisy_dir = os.path.join(args.test_dir, noisy_dir)
#         output_dir = os.path.join(args.save_dir, os.path.split(noisy_dir)[-1])
#
#         if not os.path.exists(args.save_dir):
#             os.mkdir(args.save_dir)
#             os.mkdir(output_dir)
#     # # #增强算法1处理
#         enhanced1(model_path = args.model_path, noisy_dir = noisy_dir,
#                save_tracks = args.save_tracks, saved_dir = output_dir)
#
#     # #得到增强算法1的评价指标
#         sisnr_evaluation( noisy_dir=noisy_dir, clean_dir = '/home/dataset/THCHS-30/data_thchs30/dev/',
#                saved_dir=output_dir)
    # noisy_dir_list = [d for d in os.listdir(args.test_dir) if os.path.isdir(os.path.join(args.test_dir, d))]
    # for noisy_dir in noisy_dir_list:
    #     # noisy_dir = '/home/xyj/datasets/chinese/dev_noisy_db/SNR_-3dB'
    #     noisy_dir = os.path.join(args.test_dir, noisy_dir)
    #     output_dir = os.path.join(args.save_dir, os.path.split(noisy_dir)[-1])
    #     noisy_evaluation( noisy_dir=noisy_dir, clean_dir = '/home/dataset/THCHS-30/data_thchs30/test',
    #            saved_dir=output_dir)

