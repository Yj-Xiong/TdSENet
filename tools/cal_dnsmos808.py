# from compute_metrics import *
import json
import os

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
from natsort import natsorted
from tqdm import tqdm


class ComputeScore:
    def __init__(self, primary_model_path='/home/xyj/Experience/CMG-v1/src/scores/dnsmos/DNSMOS/sig_bak_ovr.onnx', p808_model_path='/home/xyj/Experience/CMG-v1/src/scores/dnsmos/DNSMOS/model_v8.onnx'):
        self.onnx_sess = ort.InferenceSession(primary_model_path)
        self.p808_onnx_sess = ort.InferenceSession(p808_model_path)

    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length,
                                                  n_mels=n_mels)
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    # def get_polyfit_val(self, sig, bak, ovr):
    #     p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
    #     p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
    #     p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
    #
    #     sig_poly = p_sig(sig)
    #     bak_poly = p_bak(bak)
    #     ovr_poly = p_ovr(ovr)
    #
    #     return sig_poly, bak_poly, ovr_poly

    def cal_mos(self, audio, sampling_rate):
        # if audio.dtype != np.float64:
        #     raise ValueError(f"audio 类型错误: {audio.dtype}")
        fs = sampling_rate
        actual_length = len(audio)
        INPUT_LENGTH = 9.01
        len_samples = int(INPUT_LENGTH * fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / fs) - INPUT_LENGTH) + 1
        hop_len_samples = fs
        # predicted_mos_sig_seg_raw = []
        # predicted_mos_bak_seg_raw = []
        # predicted_mos_ovr_seg_raw = []
        # predicted_mos_sig_seg = []
        # predicted_mos_bak_seg = []
        # predicted_mos_ovr_seg = []
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx * hop_len_samples): int((idx + INPUT_LENGTH) * hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype('float32')[np.newaxis, :]
            p808_input_features = np.array(self.audio_melspec(audio=audio_seg[:-160])).astype('float32')[np.newaxis, :,
                                  :]
            oi = {'input_1': input_features}
            p808_oi = {'input_1': p808_input_features}
            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            # mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            # mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(mos_sig_raw, mos_bak_raw, mos_ovr_raw)
            # predicted_mos_sig_seg_raw.append(mos_sig_raw)
            # predicted_mos_bak_seg_raw.append(mos_bak_raw)
            # predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            # predicted_mos_sig_seg.append(mos_sig)
            # predicted_mos_bak_seg.append(mos_bak)
            # predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)

        #
        # COVL = np.mean(predicted_mos_ovr_seg)
        # CSIG = np.mean(predicted_mos_sig_seg)
        # CBAK = np.mean(predicted_mos_bak_seg)
        P808_MOS = np.mean(predicted_p808_mos)
        # results = [COVL, CSIG, CBAK, P808_MOS]
        return [P808_MOS]
    def cal_dnsmos808(self,file_path, save_path):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        audio_list = os.listdir(file_path)
        audio_list = [audio for audio in audio_list if audio.endswith('.wav')]
        audio_list = natsorted(audio_list)

        metrics_total = np.zeros(1)
        num = len(audio_list)

        # 用于保存每个指标的单次结果，计算标准差
        metrics_list = []
        for audio in tqdm(audio_list, desc="Evaluating DNSMOS"):
            est_path = os.path.join(file_path, audio)
            est_audio, sr = sf.read(est_path)


            assert sr == 16000

            metrics = self.cal_mos(est_audio, sr)  # 使用实例调用方法
            # metrics = np.array(metrics)

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

        print(f'P808_MOS: {metrics_avg[0]:.4f} ± {metrics_std[0]:.4f}')
        result_file_path = os.path.join(save_path, 'dnsmos_avg.json')
        results = {
            'P808_MOS': float(metrics_avg[0]),
            'P808_MOS_std': float(metrics_std[0]),
        }
        # # 打印平均值和标准差
        # print(f'COVL: {metrics_avg[0]:.4f} ± {metrics_std[0]:.4f}')
        # print(f'CSIG: {metrics_avg[1]:.4f} ± {metrics_std[1]:.4f}')
        # print(f'CBAK: {metrics_avg[2]:.4f} ± {metrics_std[2]:.4f}')
        # print(f'P808_MOS: {metrics_avg[3]:.4f} ± {metrics_std[3]:.4f}')
        result_file_path = os.path.join(save_path, 'dnsmos_avg.json')
        # results = {
        #     'COVL': float(metrics_avg[0]),
        #     'CSIG': float(metrics_avg[1]),
        #     'CBAK': float(metrics_avg[2]),
        #     'P808_MOS': float(metrics_avg[3]),
        #     'COVL_std': float(metrics_std[0]),
        #     'CSIG_std': float(metrics_std[1]),
        #     'CBAK_std': float(metrics_std[2]),
        #     'P808_MOS_std': float(metrics_std[3]),
        # }
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
if __name__ == '__main__':

    file_path = '/home/xyj/Enh_audio/Enh_audio-td_c1221/'
    save_path = '/home/xyj/Enh_audio/Enh_audio-td_c1221/'
    ComputeScore().cal_dnsmos808(file_path, save_path)