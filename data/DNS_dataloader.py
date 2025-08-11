# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 14:57:00 2021

@author: xiaohuaile
"""
import os

import librosa
# from wavinfo import WavInfoReader
import numpy as np
import soundfile as sf
import torch
from scipy import signal

'''
TRAIN_DIR: DNS data
RIR_DIR: Room impulse response
'''
TRAIN_DIR = '/home/dataset/DNS-Challenge/DNS-Challenge/datasets'
RIR_DIR = None #'/dataset/RIR_database/impulse_responses/'


#FIR, frequencies below 60Hz will be filtered
fir = signal.firls(1025,[0,40,50,60,70,8000],[0,0,0.1,0.5,1,1],fs = 16000)

def add_pyreverb(clean_speech, rir):
    '''
    convolve RIRs to the clean speech to generate reverbrant speech
    '''
    l = len(rir)//2
    reverb_speech = signal.fftconvolve(clean_speech, rir, mode="full")
    # make reverb_speech same length as clean_speech
    reverb_speech = reverb_speech[l : clean_speech.shape[0]+l]

    return reverb_speech
#按照snr混合音频
def mk_mixture(s1,s2,snr,eps = 1e-8):
    '''
    make mixture from s1 and s2 with snr
    '''
    norm_sig1 = s1 / np.sqrt(np.sum(s1 ** 2) + eps) 
    norm_sig2 = s2 / np.sqrt(np.sum(s2 ** 2) + eps)
    alpha = 10**(snr/20)
    mix = norm_sig2 + alpha*norm_sig1
    M = max(np.max(abs(mix)),np.max(abs(norm_sig2)),np.max(abs(alpha*norm_sig1))) + eps
    mix = mix / M
    norm_sig1 = norm_sig1 * alpha/ M
    norm_sig2 = norm_sig2 / M

    return norm_sig1,norm_sig2,mix,snr


class data_generator():
    def __init__(self, train_dir=TRAIN_DIR,
                 batch_size=16,
                 RIR_dir=RIR_DIR,
                 validation_rate=0.1,
                 length_per_sample=10,
                 fs=16000,
                 n_fft=400,
                 n_hop=100,
                 total_hours=3000,  # 新增参数: 总训练时长
                 add_reverb=False,
                 reverb_rate=0.5):

        self.train_dir = train_dir
        self.clean_dir = os.path.join(train_dir, 'clean')
        self.noise_dir = os.path.join(train_dir, 'noise')

        self.fs = fs
        self.batch_size = batch_size
        self.length_per_sample = length_per_sample
        self.L = length_per_sample * self.fs
        self.points_per_sample = ((self.L - n_fft) // n_hop) * n_hop + n_fft

        self.validation_rate = validation_rate
        self.add_reverb = add_reverb
        self.reverb_rate = reverb_rate

        # 计算需要生成的样本数量
        total_samples = int((total_hours * 3600) / length_per_sample)
# 3000小时转换为秒然后除以每个样本的时长
        print(f"Total samples required for {total_hours} hours of training: {total_samples}")

        # RIR 的处理逻辑
        if RIR_dir is not None:
            self.rir_dir = RIR_dir
            self.rir_list = librosa.util.find_files(self.rir_dir, ext='wav')[:total_samples]
            np.random.shuffle(self.rir_list)
            self.rir_list = self.rir_list[:total_samples]
            print('There are {} RIR clips\n'.format(len(self.rir_list)))
            self.train_rir = self.rir_list[:self.train_length]
            self.valid_rir = self.rir_list[self.train_length: self.train_length + self.valid_length]

        self.noise_file_list = os.listdir(self.noise_dir)
        self.clean_file_list = os.listdir(self.clean_dir)

        np.random.shuffle(self.clean_file_list)  # 随机打乱清洁文件列表
        self.train_length = int(len(self.clean_file_list) * (1 - validation_rate))
        self.noise_length = int(len(self.noise_file_list)* (1 - validation_rate))
        self.train_list, self.validation_list, self.noise_list = self.generating_train_validation(self.train_length,self.noise_length)

        # 对 train_list 和 validation_list 进行重复以达到 total_samples
        while len(self.train_list) < total_samples:
            repeats = ( total_samples// self.train_length) + 1  # 计算需要重复的次数
            self.train_list = np.tile(self.train_list, repeats)[:total_samples]
            # self.train_list += self.train_list[:total_samples - len(self.train_list)]

        while len(self.validation_list) < int(total_samples * self.validation_rate):
            self.validation_list += self.validation_list[
                                    :int(total_samples * self.validation_rate) - len(self.validation_list)]
        while len(self.noise_list) < total_samples:
            repeats = ( total_samples// self.noise_length) + 1  # 计算需要重复的次数
            self.noise_list = np.tile(self.noise_list, repeats)[:total_samples]

        print('Generated DNS training list...\n')
        print('There are {} samples for training, {} for validation'.format(len(self.train_list),
                                                                            len(self.validation_list)))

    # 其他方法不变

    def find_files(self,file_name):
        '''
        from file_name find parallel noise file and noisy file
        e.g.
        file_name: clean_fileid_1.wav
        noise_file_name: noise_fileid_1.wav
        noisy_file_name: noisy_fileid_1.wav
        '''
        #noise_file_name = np.random.choice(self.noise_file_list) #randomly selection
        id = file_name[:-4]
        noise_file_name = 'noise' + '_' + id
        noisy_file_name = 'noisy' + '_' + id

        # random segmentation
        Begin_S = int(np.random.uniform(0,30 - self.length_per_sample)) * self.fs
        Begin_N = int(np.random.uniform(0,30 - self.length_per_sample)) * self.fs
        return noise_file_name,noisy_file_name,Begin_S,Begin_N

    def generating_train_validation(self,training_length,noise_length):
        '''
        get training and validation data
        '''
        np.random.shuffle(self.clean_file_list)
        np.random.shuffle(self.noise_file_list)
        self.train_list,self.validation_list,self.noise_list = (self.clean_file_list[:training_length],self.clean_file_list[training_length:]
                                                                ,self.noise_file_list[:noise_length])

        return self.train_list,self.validation_list,self.noise_list

    def generator(self, batch_size, validation = False):
        '''
        data generator,
            validation: if True, get validation data genertor
        '''
        noise_data = self.noise_list
        if validation:
            train_data = self.validation_list
            # train_rir = self.valid_rir
        else:
            train_data = self.train_list
            # train_rir = self.train_rir

        np.random.shuffle(train_data)
        # np.random.shuffle(train_rir)
        np.random.shuffle(self.noise_file_list)

        N_batch = len(train_data) // batch_size
        batch_num = 0
        while (True):

            batch_clean = np.zeros([batch_size,self.points_per_sample],dtype = np.float32)
            batch_noisy = np.zeros([batch_size,self.points_per_sample],dtype = np.float32)

            for i in range(batch_size):
                # random amplitude gain
                gain = np.random.normal(loc=-5,scale=10)
                gain = 10**(gain/10)
                gain = min(gain,3)
                gain = max(gain,0.01)

                SNR = np.random.uniform(-5,15)
                sample_num = batch_num*batch_size + i
                #get the path of clean audio
                clean_f = train_data[sample_num]
                # rir_f = train_rir[sample_num]
                reverb_rate = np.random.rand()
                noise_f = noise_data[sample_num]
                noise_file, noisy_f, Begin_S,Begin_N = self.find_files(clean_f)
                clean_s = sf.read(os.path.join(self.clean_dir,clean_f),dtype = 'float32',start= Begin_S,stop = Begin_S + self.points_per_sample)[0]
                noise_s = sf.read(os.path.join(self.noise_dir,noise_f),dtype = 'float32',start= Begin_N,stop = Begin_N + self.points_per_sample)[0]
                # print(f"clean_f: {clean_f}, noise_f: {noise_f}, noisy_f: {noisy_f}")
                # 确保 noise_s 和 clean_s 都不是空数组
                if noise_s is None or clean_s is None:
                    raise ValueError("noise_s or clean_s is None. Check your data loading logic.")
                # 确保 noise_s 和 clean_s 都不是空数组
                if noise_s.size == 0 or clean_s.size == 0:
                    raise ValueError("noise_s or clean_s is empty. Check your data loading logic.")

                # 对齐 noise_s 和 clean_s 的长度
                if noise_s.size > clean_s.size:
                    noise_s = noise_s[:clean_s.size]  # 截断多余的部分
                elif noise_s.size < clean_s.size:
                    if noise_s.size == 0:
                        raise ValueError("noise_s is empty. Cannot repeat an empty array.")
                    repeats = (clean_s.size // noise_s.size) + 1  # 计算需要重复的次数
                    noise_s = np.tile(noise_s, repeats)[:clean_s.size]  # 生成重复后的噪声并截取到相同长度

                # 确保长度一致
                assert noise_s.size == clean_s.size, "noise_s and clean_s must have the same length after alignment."

                # 调用 add_pyreverb
                clean_s = add_pyreverb(clean_s, fir)

                #noise_s = noise_s - np.mean(noise_s)
                # if self.add_reverb:
                #     if reverb_rate < self.reverb_rate:
                #         rir_s = sf.read(rir_f,dtype = 'float32')[0]
                #         if len(rir_s.shape)>1:
                #             rir_s = rir_s[:,0]
                #         clean_s = add_pyreverb(clean_s, rir_s)

                clean_s,noise_s,noisy_s,_ = mk_mixture(clean_s,noise_s,SNR,eps = 1e-8)

                batch_clean[i,:] = clean_s * gain
                batch_noisy[i,:] = noisy_s * gain

            batch_num += 1

            if batch_num == N_batch:
                batch_num = 0

            # Convert to torch tensors
            batch_clean_tensor = torch.tensor(batch_clean)
            batch_noisy_tensor = torch.tensor(batch_noisy)
            return batch_clean_tensor, batch_noisy_tensor
            # 使用 torch.stack 进行堆叠
            # return torch.stack((batch_clean_tensor, batch_noisy_tensor), dim=0)


# 假设你的代码在此处
# 实例化数据生成器
def main():
    train_ds = data_generator(train_dir='/home/dataset/DNS-Challenge/DNS-Challenge/datasets/',
                                         batch_size=2,
                                         RIR_dir=None,
                                         validation_rate=0.,
                                         length_per_sample=10,
                                         fs=16000,
                                         n_fft=400,
                                         n_hop=100,
                                         total_hours=3000,  # 新增参数: 总训练时长
                                         add_reverb=False,
                                         reverb_rate=0.5)

# 创建生成器对象
    generator = train_ds.generator(1, validation=False)  # 或 validation=True 用于验证数据

# 获取一定数量的批次数据
    for batch in range(10):  # 可以根据需要迭代更多的批次
        batch_clean, batch_noisy = next(generator)  # 获取下一个批次
        print(f'Batch {batch}:')
        print('Clean batch shape:', batch_clean.shape)
        print('Noisy batch shape:', batch_noisy.shape)
