import os
from argparse import ArgumentParser
from glob import glob
from os.path import join

import numpy as np
from librosa import load
from soundfile import write
from tqdm import tqdm

# Params
min_snr = -15
step = 5
max_snr = 0
sr = 16000  # sample rate

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--clean", type=str, help='path to clean',
                        default='/home/dataset/Voicebank/noisy-vctk-16k/clean_testset_wav_16k/')
    parser.add_argument("--noise", type=str, help='path to noise',
                        default='/home/xyj/NoiseX-92/babble.wav')
    parser.add_argument("--target", type=str, help='target path for training files',
                        default='/home/xyj/datasets/english/vb_babble')
    args = parser.parse_args()

    # Initialize seed for re-producability
    np.random.seed(0)

    # Clean speech
    clean_files = sorted(glob(args.clean + '/*.wav', recursive=True))

    with open('/home/dataset/THUYG-20/list/cv.list', 'r') as f:
        lines = f.readlines()
        clean_files = [join(args.clean, line.replace('\n', '').split('/')[-1]) for line in lines]
        f.close()

    snr_dB = list(range(min_snr, max_snr + step, step))

    if os.path.isdir(args.noise):
        noise_files = glob(args.noise + '/babble.wav', recursive=True)
        # for clean_file in tqdm(clean_files):
        #     s, sr_c = load(clean_file, sr=sr)
        #
        #     snr_dB = np.random.uniform(min_snr, max_snr)
        #     noise_ind = np.random.randint(len(noises))
        #     speech_power = 1 / len(s) * np.sum(s ** 2)
        #
        #     n = noises[noise_ind]
        #     start = np.random.randint(len(n) - len(s))
        #     n = n[start:start + len(s)]
        #
        #     noise_power = 1 / len(n) * np.sum(n ** 2)
        #     noise_power_target = speech_power * np.power(10, -snr_dB / 10)
        #     k = noise_power_target / noise_power
        #     n = n * np.sqrt(k)
        #     x = s + n
        #
        #     file_name = speech_file.split('/')[-1]
        #     write(os.path.join(args.target, file_name), x, sr)
    else:
        n, _ = load(args.noise, sr=sr)

        for snr in snr_dB:
            print(f'Mix clean and noise with {str(snr)}dB')
            for clean_file in tqdm(clean_files):
                s, _ = load(clean_file, sr=sr)

                start = np.random.randint(len(n) - len(s))
                n_slice = n[start:start + len(s)]

                clean_power = 1 / len(s) * np.sum(s ** 2)
                noise_power = 1 / len(n_slice) * np.sum(n_slice ** 2)

                noise_power_target = clean_power * np.power(10, -snr / 10)
                k = noise_power_target / noise_power
                n_slice = n_slice * np.sqrt(k)
                x = s + n_slice

                file_name = clean_file.split('/')[-1]
                target_path = os.path.join(args.target, f'{str(snr)}dB')
                if not os.path.exists(target_path):
                    os.mkdir(target_path)
                write(os.path.join(target_path, file_name), x, sr)

            print('Done!')
