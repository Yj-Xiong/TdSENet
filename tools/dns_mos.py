import argparse

# URL for the web service
SCORING_URI_DNSMOS = 'https://dnsmos.azurewebsites.net/score'
SCORING_URI_DNSMOS_P835 = 'https://dnsmos.azurewebsites.net/v1/dnsmosp835/score'
# If the service is authenticated, set the key or token
AUTH_KEY = 'd3VoYW4tdW5pdjpkbnNtb3M='

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Basic {AUTH_KEY}'


import os
import glob
import json
import requests
import soundfile as sf
import librosa
import pandas as pd
import numpy as np

def main(args):
    print(args.testset_dir)
    audio_clips_list = glob.glob(os.path.join(args.testset_dir, "*.wav"))  # glob：搜索列表中符合的文件，返回列表
    print(audio_clips_list)
    scores = []
    dir_path = args.score_file.split('score.csv')[0]
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if not os.path.exists(os.path.join(dir_path, 'file_mos.txt')):
        f = open(os.path.join(dir_path, 'file_mos.txt'), 'w')
        dict = {}
    else:
        f = open(os.path.join(dir_path, 'file_mos.txt'), 'r')
        dict = {}
        lines = f.readlines()
        for line in lines:
            utt_id = line.split('.wav')[0]
            dict[utt_id] = 1
    flag = 0
    for fpath in audio_clips_list:
        utt_id = fpath.split('\\')[-1].split('.wav')[0]
        if utt_id in dict:
            print('find uttid', utt_id)
            continue
        flag = 1
        f = open(os.path.join(dir_path, 'file_mos.txt'), 'a+')
        audio, fs = sf.read(fpath)
        if fs != 16000:
            print('Resample to 16k')
            audio = librosa.resample(audio, orig_sr=fs, target_sr=16000)
        data = {"data": audio.tolist(), "filename": os.path.basename(fpath)}
        input_data = json.dumps(data)
        if args.method == 'p808':
            u = SCORING_URI_DNSMOS
        else:
            u = SCORING_URI_DNSMOS_P835
        try_flag = 1
        while try_flag:
            try:
                resp = requests.post(u, data=input_data, headers=headers, timeout=60)  # 增加超时时间
                resp.raise_for_status()  # 检查请求是否成功
                score_dict = resp.json()
                try_flag = 0
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
                print('retry_1')
                continue
            try:
                score_dict['file_name'] = os.path.basename(fpath)
                if args.method == 'p808':
                    f.write(score_dict['file_name'] + ' ' + str(score_dict['mos']) + '\n')
                    print(score_dict['mos'], ' ', score_dict['file_name'])
                else:
                    f.write(score_dict['file_name'] + ' SIG[{}], BAK[{}], OVR[{}]'.format(score_dict['mos_sig'],
                                                                                          score_dict['mos_bak'],
                                                                                          score_dict['mos_ovr']) + '\n')
                    print(score_dict['file_name'] + ' SIG[{}], BAK[{}], OVR[{}]'.format(score_dict['mos_sig'],
                                                                                        score_dict['mos_bak'],
                                                                                        score_dict['mos_ovr']))
                try_flag = 0
            except Exception as e:
                print(f"Processing response failed: {e}")
                print('retry_2')
                continue
        f.close()
        scores.append(score_dict)
    if flag:
        df = pd.DataFrame(scores)
        if args.method == 'p808':
            print('Mean MOS Score for the files is ', np.mean(df['mos']))
        else:
            print('Mean scores for the files: SIG[{}], BAK[{}], OVR[{}]'.format(np.mean(df['mos_sig']),
                                                                                np.mean(df['mos_bak']),
                                                                                np.mean(df['mos_ovr'])))

        if args.score_file:
            df.to_csv(args.score_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset_dir",
                        default=r'/home/dataset/DNS-Challenge/DNS-Challenge/datasets/test_set/synthetic/no_reverb/noisy',
                        help='Path to the dir containing audio clips to be evaluated')
    parser.add_argument('--score_file', default=r'/home/xyj/Enh_audio/CMGAN-multi/Voicebank/831_HfCB-U_39_DNS/',
                        help='If you want the scores in a CSV file provide the full path')
    parser.add_argument('--method', default='p835', const='p808', nargs='?', choices=['p808', 'p835'],
                        help='Choose which method to compute P.808 or P.835. Default is P.808')
    args = parser.parse_args()
    main(args)
