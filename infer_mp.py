import argparse
import os
import json,sys,glob
import numpy as np
import soundfile as sf
import torch
import torchaudio
from env import AttrDict
from natsort import natsorted
from models import generator_primek as generator
# from models import Td_SENet as generator
from tools.gtcrn_compute_metrics import *
from utils import *
def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]
def mag_pha_stft(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    hann_window = torch.hann_window(win_size).to(y.device)
    stft_spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                           center=center, pad_mode='reflect', normalized=False, return_complex=True)
    stft_spec = torch.view_as_real(stft_spec)
    mag = torch.sqrt(stft_spec.pow(2).sum(-1) + (1e-9))
    pha = torch.atan2(stft_spec[:, :, :, 1] + (1e-10), stft_spec[:, :, :, 0] + (1e-5))
    # Magnitude Compression
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)

    return mag, pha, com


def mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    # Magnitude Decompression
    mag = torch.pow(mag, (1.0 / compress_factor))
    com = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
    hann_window = torch.hann_window(win_size).to(com.device)
    wav = torch.istft(com, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=center)

    return wav

@torch.no_grad()
def enhance_one_track(model, audio_path, saved_dir, cut_len, n_fft=400, hop=100, save_tracks=False,device='cuda:1'):

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
    noisy_amp, noisy_pha, noisy_com = mag_pha_stft(noisy, n_fft, hop, n_fft, 0.3)
    amp_g, pha_g, com_g = model(noisy_amp, noisy_pha)
    audio_g = mag_pha_istft(amp_g, pha_g, n_fft, hop, n_fft, 0.3)


    est_audio = audio_g / c
    est_audio = torch.flatten(est_audio)[:length].cpu().numpy()
    # assert len(est_audio) == length
    if save_tracks:
        saved_path = os.path.join(saved_dir, name)
        sf.write(saved_path, est_audio, sr)

    return est_audio, length


def enhanced1(model_path, noisy_dir, save_tracks, saved_dir,device='cuda:1'):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)
    torch.device(device)
    map_location = torch.device(device)
    n_fft = 400
    model = generator.LKFCA_Net(h,num_tsblock=4).to(device)
    print('Loading model from {}'.format(model_path))
    state_dict = torch.load(model_path,map_location=map_location)
    model.load_state_dict(state_dict['generator'])
    # model.load_state_dict()
    model.eval()
    print('Complete loading.')
    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)

    audio_list = os.listdir(noisy_dir)

    # 仅保留以 .wav 结尾的文件
    audio_list = [audio for audio in audio_list if audio.endswith('.wav')]
    audio_list = natsorted(audio_list)
    num = len(audio_list)
    for i, audio in enumerate(audio_list):
        noisy_path = os.path.join(noisy_dir, audio)
        est_audio, length = enhance_one_track(model, noisy_path, saved_dir, 16000 * 6, n_fft, n_fft // 4, save_tracks)
        # 显示当前进度百分比
        progress_percentage = (i + 1) / num * 100  # 计算百分比
        sys.stdout.write(f'\rEnhancement Processing {i + 1}/{num} ({progress_percentage:.2f}%): {audio}')
        sys.stdout.flush()  # 刷新输出

    print()  # 确保最后一行打印后换行


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

    for i, audio in enumerate(audio_list):
        est_path = os.path.join(saved_dir, audio)
        est_audio, sr = sf.read(est_path)

        clean_path = os.path.join(clean_dir, audio)
        clean_audio, sr = sf.read(clean_path)

        assert sr == 16000
        metrics = compute_all_metrics(clean_audio, est_audio, sr, 0)
        metrics = np.array(metrics)

        metrics_total += metrics
        metrics_list.append(metrics)  # 添加指标到列表

        # 显示当前进度百分比
        progress_percentage = (i + 1) / num * 100
        sys.stdout.write(f'\rEvaluation Processing {i + 1}/{num} ({progress_percentage:.2f}%): {audio}')
        sys.stdout.flush()

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
    for i, audio in enumerate(audio_list):
        est_path = os.path.join(noisy_dir, audio)
        est_audio, sr = sf.read(est_path)

        clean_path = os.path.join(clean_dir, audio)
        clean_audio, sr = sf.read(clean_path)

        assert sr == 16000
        metrics = compute_all_metrics(clean_audio, est_audio, sr, 0)
        metrics = np.array(metrics)

        metrics_total += metrics
        metrics_list.append(metrics)  # 添加指标到列表

        # 显示当前进度百分比
        progress_percentage = (i + 1) / num * 100
        sys.stdout.write(f'\rEvaluation Processing {i + 1}/{num} ({progress_percentage:.2f}%): {audio}')
        sys.stdout.flush()

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
    # result_file_path = os.path.join(saved_dir, 'metrics_avg.json')
    # results = {
    #     'pesq': metrics_avg[0],
    #     'csig': metrics_avg[1],
    #     'cbak': metrics_avg[2],
    #     'covl': metrics_avg[3],
    #     'ssnr': metrics_avg[4],
    #     'sisnr': metrics_avg[5],
    #     'sisdr': metrics_avg[6],
    #     'stoi': metrics_avg[7]
    # }

    # 如果文件存在，加载已有数据并追加新结果
    # if os.path.exists(result_file_path):
    #     with open(result_file_path, 'r') as json_file:
    #         all_results = json.load(json_file)
    # else:
    #     all_results = []

    # 追加当前实验的结果
    # all_results.append(results)

    # 保存更新后的结果
    # with open(result_file_path, 'w') as json_file:
    #     json.dump(all_results, json_file, indent=4)
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default='/home/xyj/Experience/PrimeK-Net-main/g_best_51epoch',
                    help="the path where the model is saved")

parser.add_argument("--test_dir", type=str, default='/home/dataset/Uighur-Chinese-English/Chinese/test/babble/',
                    help="noisy tracks dir to be enhanced")
parser.add_argument("--save_tracks", type=str, default=True, help="save predicted tracks or not")
parser.add_argument("--save_dir", type=str, default='../Enh_audio-primeknet-chinese-babble', help="where enhanced tracks to be saved")
args = parser.parse_args()


if __name__ == '__main__':
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    # # 列出当前目录下的目录名称列表
    noisy_dir_list = [d for d in os.listdir(args.test_dir) if os.path.isdir(os.path.join(args.test_dir, d))]

    for noisy_dir in noisy_dir_list:
        noisy_dir = os.path.join(args.test_dir, noisy_dir)
        output_dir = os.path.join(args.save_dir, os.path.split(noisy_dir)[-1])

        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
            os.mkdir(output_dir)
    # # #增强算法1处理
    #     enhanced1(model_path = args.model_path, noisy_dir = noisy_dir,
    #            save_tracks = args.save_tracks, saved_dir = output_dir)

    # #得到增强算法1的评价指标
        sisnr_evaluation( noisy_dir=noisy_dir, clean_dir = '/home/dataset/THCHS-30/data_thchs30/test/',
               saved_dir=output_dir)
# parser = argparse.ArgumentParser()


# parser.add_argument("--model_path", type=str, default='/home/xyj/Experience/MP-SENet/cp_model_C/g_best_77epoch',
#                     help="the path where the model is saved")
# # parser.add_argument("--model_path", type=str, default='/home/xyj/Experience/CMG-v1/src/ckpt_524-w|o_FFB/CMGAN_epoch_44_0.056_0.003',
# #                     help="the path where the model is saved")
# # chinese :/home/wrl/new_torch/exp_result_audio/model/CMGAN-chinese/CMGAN_epoch_34_6.056
# # chinese-multi: /home/wrl/new_torch/exp_result_audio/model/CMGAN-chinese-multi/CMGAN_epoch_85_8.081_0.002_2.013
# parser.add_argument("--test_dir", type=str, default='/home/dataset/THCHS-30/data_thchs30/test',
#                     help="noisy tracks dir to be enhanced")
# parser.add_argument("--save_tracks", type=str, default=True, help="save predicted tracks or not")
# parser.add_argument("--save_dir", type=str, default='../Enh_audio-Mp—C', help="where enhanced tracks to be saved")
# args = parser.parse_args()
#
#
# if __name__ == '__main__':
#
#
#     # noisy_dir = os.path.join(args.test_dir, 'noisy_testset_wav_16k')
#     # clean_dir = os.path.join(args.test_dir, 'clean_testset_wav_16k')
#     noisy_dir = '/home/xyj/datasets/chinese/test_noisy'
#     # noisy_dir = os.path.join(args.test_dir, noisy_dir)
#     if not os.path.exists(args.save_dir):
#         os.mkdir(args.save_dir)
#             # os.mkdir(output_dir)
#     #增强算法1处理
#     # enhanced1(model_path = args.model_path, noisy_dir = noisy_dir,
#     #            save_tracks = args.save_tracks, saved_dir = args.save_dir)
#
#     # 得到增强算法1的评价指标
#     sisnr_evaluation( noisy_dir=noisy_dir, clean_dir = '/home/dataset/THCHS-30/data_thchs30/test/',
#                saved_dir=args.save_dir)
# # parser = argparse.ArgumentParser()
# # parser.add_argument("--model_path", type=str, default='/home/xyj/Experience/CMG-v1/src/ckpt/tdsenet_0.01cp/epoch58-pesq:3.568-loss_sum:0.5103:-g:0.505-d:0.004',
# #                     help="the path where the model is saved")
# #
# # parser.add_argument("--test_dir", type=str, default='/home/dataset/Uighur-Chinese-English/Chinese/test/babble/',
# #                     help="noisy tracks dir to be enhanced")
# # parser.add_argument("--save_tracks", type=str, default=True, help="save predicted tracks or not")
# # parser.add_argument("--save_dir", type=str, default='../Enh_audio-tdse_chinese', help="where enhanced tracks to be saved")
# # args = parser.parse_args()
# #
# #
# # if __name__ == '__main__':
# #     # if not os.path.exists(args.save_dir):
# #     #     os.mkdir(args.save_dir)
# #     #
# #     # # # 列出当前目录下的目录名称列表
# #     # noisy_dir_list = [d for d in os.listdir(args.test_dir) if os.path.isdir(os.path.join(args.test_dir, d))]
# #     #
# #     # for noisy_dir in noisy_dir_list:
# #     #     noisy_dir = os.path.join(args.test_dir, noisy_dir)
# #     #     output_dir = os.path.join(args.save_dir, os.path.split(noisy_dir)[-1])
# #     #
# #     #     if not os.path.exists(args.save_dir):
# #     #         os.mkdir(args.save_dir)
# #     #         os.mkdir(output_dir)
# #     # # #增强算法1处理
# #     #     enhanced1(model_path = args.model_path, noisy_dir = noisy_dir,
# #     #            save_tracks = args.save_tracks, saved_dir = output_dir)
# #     #
# #     # # #得到增强算法1的评价指标
# #     #     sisnr_evaluation( noisy_dir=noisy_dir, clean_dir = '/home/dataset/THCHS-30/data_thchs30/test/',
# #     #            saved_dir=output_dir)
# #     # for noisy_dir in noisy_dir_list:
# #     noisy_dir = '/home/xyj/datasets/uyghur/test_noisy'
# #     # noisy_dir = os.path.join(args.test_dir, noisy_dir)
# #     output_dir = os.path.join(args.save_dir, os.path.split(noisy_dir)[-1])
# #     noisy_evaluation( noisy_dir=noisy_dir, clean_dir = '/home/dataset/voicebank_48kHz/noisy_testset_wav/',
# #                saved_dir=output_dir)
# #
