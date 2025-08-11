import os
import random

import torch.utils.data
import torchaudio
from natsort import natsorted
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
from utils import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class DNSDataset(torch.utils.data.Dataset):
    def __init__(self, clean_dir,noisy_dir, file_list,cut_len=16000 * 2, ):
        self.cut_len = cut_len
        self.clean_dir = os.path.join(clean_dir, 'clean')
        self.noisy_dir = os.path.join(noisy_dir, 'SNR=[-5,-15]')
        if not os.path.exists(self.clean_dir) or not os.path.exists(self.noisy_dir):
            raise ValueError("Invalid directory path")

        # 检查文件列表路径是否有效
        if not os.path.exists(file_list):
            raise ValueError("Invalid file list path")

        # 读取文件列表
        try:
            with open(file_list, 'r') as f:
                self.clean_wav_name = f.read().splitlines()
        except OSError as e:
            raise OSError(f"Failed to open file list: {e}")

        # 读取文件列表
        with open(file_list, 'r') as f:
            self.clean_wav_name = f.read().splitlines()

        # 确保文件列表是有序的
        self.clean_wav_name = natsorted(self.clean_wav_name)

    def __len__(self):
        return len(self.clean_wav_name)

    def __getitem__(self, idx):
        clean_file = os.path.join(self.clean_dir, self.clean_wav_name[idx])
        noisy_file = os.path.join(self.noisy_dir, self.clean_wav_name[idx])

        clean_ds, _ = torchaudio.load(clean_file)
        noisy_ds, _ = torchaudio.load(noisy_file)
        clean_ds = clean_ds.squeeze()
        noisy_ds = noisy_ds.squeeze()
        length = len(clean_ds)
        assert length == len(noisy_ds)

        if length < self.cut_len:
            # 如果音频长度小于指定的切片长度，进行重复拼接
            units = self.cut_len // length
            clean_ds_final = []
            noisy_ds_final = []
            for _ in range(units):
                clean_ds_final.append(clean_ds)
                noisy_ds_final.append(noisy_ds)
            clean_ds_final.append(clean_ds[: self.cut_len % length])
            noisy_ds_final.append(noisy_ds[: self.cut_len % length])
            clean_ds = torch.cat(clean_ds_final, dim=-1)
            noisy_ds = torch.cat(noisy_ds_final, dim=-1)
        else:
            # 随机剪切2秒的音频片段
            wav_start = random.randint(0, length - self.cut_len)
            clean_ds = clean_ds[wav_start:wav_start + self.cut_len]
            noisy_ds = noisy_ds[wav_start:wav_start + self.cut_len]

        return clean_ds, noisy_ds, length

class DemandDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, cut_len=16000*2):
        self.cut_len = cut_len
        self.clean_dir = os.path.join(data_dir, 'clean_trainset_28spk_wav_16k')
        self.noisy_dir = os.path.join(data_dir, 'noisy_trainset_28spk_wav_16k')
        self.clean_wav_name = os.listdir(self.clean_dir)
        self.clean_wav_name = natsorted(self.clean_wav_name)

    def __len__(self):
        return len(self.clean_wav_name)

    def __getitem__(self, idx):
        clean_file = os.path.join(self.clean_dir, self.clean_wav_name[idx])
        noisy_file = os.path.join(self.noisy_dir, self.clean_wav_name[idx])

        clean_ds, _ = torchaudio.load(clean_file)
        noisy_ds, _ = torchaudio.load(noisy_file)
        clean_ds = clean_ds.squeeze()
        noisy_ds = noisy_ds.squeeze()
        length = len(clean_ds)
        assert length == len(noisy_ds)
        if length < self.cut_len:
            units = self.cut_len // length
            clean_ds_final = []
            noisy_ds_final = []
            for i in range(units):
                clean_ds_final.append(clean_ds)
                noisy_ds_final.append(noisy_ds)
            clean_ds_final.append(clean_ds[: self.cut_len%length])
            noisy_ds_final.append(noisy_ds[: self.cut_len%length])
            clean_ds = torch.cat(clean_ds_final, dim=-1)
            noisy_ds = torch.cat(noisy_ds_final, dim=-1)
        else:
            # randomly cut 2 seconds segment
            wav_start = random.randint(0, length - self.cut_len)
            noisy_ds = noisy_ds[wav_start:wav_start + self.cut_len]
            clean_ds = clean_ds[wav_start:wav_start + self.cut_len]
            clean_ds = torch.stft(clean_ds, 400, 100, window=torch.hamming_window(400),onesided=True)
            clean_ds = torch.istft(clean_ds, 400, 100, window=torch.hamming_window(400),onesided=True)

        return clean_ds, noisy_ds, length



class DemandDataset_1(torch.utils.data.Dataset):
    def __init__(self, data_dir, cut_len=16000*2):
        self.cut_len = cut_len
        self.clean_dir = os.path.join(data_dir, 'clean_testset_wav_16k')
        self.noisy_dir = os.path.join(data_dir, 'noisy_testset_wav_16k')
        self.clean_wav_name = os.listdir(self.clean_dir)
        self.clean_wav_name = natsorted(self.clean_wav_name)

    def __len__(self):
        return len(self.clean_wav_name)

    def __getitem__(self, idx):
        clean_file = os.path.join(self.clean_dir, self.clean_wav_name[idx])
        noisy_file = os.path.join(self.noisy_dir, self.clean_wav_name[idx])

        clean_ds, _ = torchaudio.load(clean_file)
        noisy_ds, _ = torchaudio.load(noisy_file)
        clean_ds = clean_ds.squeeze()
        noisy_ds = noisy_ds.squeeze()
        length = len(clean_ds)
        assert length == len(noisy_ds)
        if length < self.cut_len:
            units = self.cut_len // length
            clean_ds_final = []
            noisy_ds_final = []
            for i in range(units):
                clean_ds_final.append(clean_ds)
                noisy_ds_final.append(noisy_ds)
            clean_ds_final.append(clean_ds[: self.cut_len%length])
            noisy_ds_final.append(noisy_ds[: self.cut_len%length])
            clean_ds = torch.cat(clean_ds_final, dim=-1)
            noisy_ds = torch.cat(noisy_ds_final, dim=-1)
        else:
            # randomly cut 2 seconds segment
            wav_start = random.randint(0, length - self.cut_len)
            noisy_ds = noisy_ds[wav_start:wav_start + self.cut_len]
            clean_ds = clean_ds[wav_start:wav_start + self.cut_len]

        return clean_ds, noisy_ds, length
class DNSDataset_test_with_reverb(torch.utils.data.Dataset):
    def __init__(self, data_dir, cut_len=16000*2):
        self.cut_len = cut_len
        self.clean_dir = os.path.join(data_dir, 'clean_testset_wav_16k')
        self.noise_dir = os.path.join(data_dir, 'noisy_testset_wav_16k')
        self.clean_wav_name = os.listdir(self.clean_dir)
        self.clean_wav_name = natsorted(self.clean_wav_name)

    def __len__(self):
        return len(self.clean_wav_name)

    def __getitem__(self, idx):
        clean_file = os.path.join(self.clean_dir, self.clean_wav_name[idx])
        noisy_file = os.path.join(self.noisy_dir, self.clean_wav_name[idx])

        clean_ds, _ = torchaudio.load(clean_file)
        noisy_ds, _ = torchaudio.load(noisy_file)
        clean_ds = clean_ds.squeeze()
        noisy_ds = noisy_ds.squeeze()
        length = len(clean_ds)
        assert length == len(noisy_ds)
        if length < self.cut_len:
            units = self.cut_len // length
            clean_ds_final = []
            noisy_ds_final = []
            for i in range(units):
                clean_ds_final.append(clean_ds)
                noisy_ds_final.append(noisy_ds)
            clean_ds_final.append(clean_ds[: self.cut_len%length])
            noisy_ds_final.append(noisy_ds[: self.cut_len%length])
            clean_ds = torch.cat(clean_ds_final, dim=-1)
            noisy_ds = torch.cat(noisy_ds_final, dim=-1)
        else:
            # randomly cut 2 seconds segment
            wav_start = random.randint(0, length - self.cut_len)
            noisy_ds = noisy_ds[wav_start:wav_start + self.cut_len]
            clean_ds = clean_ds[wav_start:wav_start + self.cut_len]

        return clean_ds, noisy_ds, length
class DNSDataset_test_with_reverb(torch.utils.data.Dataset):
    def __init__(self, data_dir, cut_len=16000*2):
        self.cut_len = cut_len
        self.clean_dir = os.path.join(data_dir, 'clean_testset_wav_16k')
        self.noise_dir = os.path.join(data_dir, 'noisy_testset_wav_16k')
        self.clean_wav_name = os.listdir(self.clean_dir)
        self.clean_wav_name = natsorted(self.clean_wav_name)

    def __len__(self):
        return len(self.clean_wav_name)

    def __getitem__(self, idx):
        clean_file = os.path.join(self.clean_dir, self.clean_wav_name[idx])
        noisy_file = os.path.join(self.noisy_dir, self.clean_wav_name[idx])

        clean_ds, _ = torchaudio.load(clean_file)
        noisy_ds, _ = torchaudio.load(noisy_file)
        clean_ds = clean_ds.squeeze()
        noisy_ds = noisy_ds.squeeze()
        length = len(clean_ds)
        assert length == len(noisy_ds)
        if length < self.cut_len:
            units = self.cut_len // length
            clean_ds_final = []
            noisy_ds_final = []
            for i in range(units):
                clean_ds_final.append(clean_ds)
                noisy_ds_final.append(noisy_ds)
            clean_ds_final.append(clean_ds[: self.cut_len%length])
            noisy_ds_final.append(noisy_ds[: self.cut_len%length])
            clean_ds = torch.cat(clean_ds_final, dim=-1)
            noisy_ds = torch.cat(noisy_ds_final, dim=-1)
        else:
            # randomly cut 2 seconds segment
            wav_start = random.randint(0, length - self.cut_len)
            noisy_ds = noisy_ds[wav_start:wav_start + self.cut_len]
            clean_ds = clean_ds[wav_start:wav_start + self.cut_len]

        return clean_ds, noisy_ds, length
class DNSDataset_test_reverb(torch.utils.data.Dataset):
    def __init__(self, data_dir, cut_len=16000*2):
        self.cut_len = cut_len
        self.clean_dir = os.path.join(data_dir, 'clean')
        self.noisy_dir = os.path.join(data_dir, 'noisy')
        self.clean_wav_name = os.listdir(self.clean_dir)
        self.clean_wav_name = natsorted(self.clean_wav_name)
        self.noisy_wav_name = os.listdir(self.noisy_dir)
        self.noisy_wav_name = natsorted(self.noisy_wav_name)

    def __len__(self):
        return len(self.clean_wav_name)

    def __getitem__(self, idx):
        clean_file = os.path.join(self.clean_dir, self.clean_wav_name[idx])
        noisy_file = os.path.join(self.noisy_dir, self.noisy_wav_name[idx])

        clean_ds, _ = torchaudio.load(clean_file)
        noisy_ds, _ = torchaudio.load(noisy_file)
        clean_ds = clean_ds.squeeze()
        noisy_ds = noisy_ds.squeeze()
        length = len(clean_ds)
        assert length == len(noisy_ds)
        if length < self.cut_len:
            units = self.cut_len // length
            clean_ds_final = []
            noisy_ds_final = []
            for i in range(units):
                clean_ds_final.append(clean_ds)
                noisy_ds_final.append(noisy_ds)
            clean_ds_final.append(clean_ds[: self.cut_len%length])
            noisy_ds_final.append(noisy_ds[: self.cut_len%length])
            clean_ds = torch.cat(clean_ds_final, dim=-1)
            noisy_ds = torch.cat(noisy_ds_final, dim=-1)
        else:
            # randomly cut 2 seconds segment
            wav_start = random.randint(0, length - self.cut_len)
            noisy_ds = noisy_ds[wav_start:wav_start + self.cut_len]
            clean_ds = clean_ds[wav_start:wav_start + self.cut_len]

        return clean_ds, noisy_ds, length

def load_data(ds_dir, batch_size, n_cpu, cut_len):
    torchaudio.set_audio_backend("sox_io")         # in linux
    # train_dir = os.path.join(ds_dir, 'train')
    # test_dir = os.path.join(ds_dir, 'test')

    train_ds = DemandDataset(ds_dir, cut_len)
    test_ds = DemandDataset_1(ds_dir, cut_len)

    # train_dataset = torch.utils.data.DataLoader(dataset=train_ds, batch_size=batch_size, pin_memory=True, shuffle=False,
    #                                             sampler=DistributedSampler(train_ds),
    #                                             drop_last=True, num_workers=n_cpu)
    # test_dataset = torch.utils.data.DataLoader(dataset=test_ds, batch_size=batch_size, pin_memory=True, shuffle=False,
    #                                             sampler=DistributedSampler(test_ds),
    #                                            drop_last=False, num_workers=n_cpu)

    train_dataset = torch.utils.data.DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True,
                                                drop_last=True, num_workers=n_cpu,persistent_workers=True,pin_memory=True)
    test_dataset = torch.utils.data.DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False,
                                               drop_last=False, num_workers=n_cpu,persistent_workers=True,pin_memory=True)



    # train_dataset = DataLoaderX(dataset=train_ds, batch_size=batch_size, shuffle=True, pin_memory=True,
    #                             drop_last=True, num_workers=n_cpu)
    # test_dataset = DataLoaderX(dataset=test_ds, batch_size=batch_size, shuffle=False, pin_memory=True,
    #                            drop_last=False, num_workers=n_cpu)

    return train_dataset, test_dataset
def load_dnsdata(clean_dir,noisy_dir, batch_size, n_cpu, cut_len,list):
    torchaudio.set_audio_backend("sox_io")         # in linux
    # train_dir = os.path.join(ds_dir, 'train')
    # test_dir = os.path.join(ds_dir, 'test')

    train_ds = DNSDataset(clean_dir,noisy_dir, list,cut_len,)
    # test_ds = DemandDataset_1(ds_dir, cut_len)

    # train_dataset = torch.utils.data.DataLoader(dataset=train_ds, batch_size=batch_size, pin_memory=True, shuffle=False,
    #                                             sampler=DistributedSampler(train_ds),
    #                                             drop_last=True, num_workers=n_cpu)
    # test_dataset = torch.utils.data.DataLoader(dataset=test_ds, batch_size=batch_size, pin_memory=True, shuffle=False,
    #                                             sampler=DistributedSampler(test_ds),
    #                                            drop_last=False, num_workers=n_cpu)

    train_dataset = torch.utils.data.DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True,
                                                drop_last=True, num_workers=n_cpu)

    return train_dataset

def load_dns_testdata_no_reverb(nr_dir,batch_size, n_cpu, cut_len):

    torchaudio.set_audio_backend("sox_io")         # in linux
    test_nr_ds = DNSDataset_test_reverb(nr_dir, cut_len)


    test_nr_dataset = torch.utils.data.DataLoader(dataset=test_nr_ds, batch_size=1, shuffle=False,
                                               drop_last=False, num_workers=n_cpu)

    return test_nr_dataset
def load_dns_testdata_with_reverb(wr_dir,batch_size, n_cpu, cut_len):

    torchaudio.set_audio_backend("sox_io")         # in linux
    test_wr_ds = DNSDataset_test_reverb(wr_dir, cut_len)
    test_wr_dataset = torch.utils.data.DataLoader(dataset=test_wr_ds, batch_size=batch_size, shuffle=False,
                                                  drop_last=False, num_workers=n_cpu)

    return test_wr_dataset
def load_dns_testdata(nr_dir,wr_dir,blind_dir, batch_size, n_cpu, cut_len):
    torchaudio.set_audio_backend("sox_io")         # in linux
    # train_dir = os.path.join(ds_dir, 'train')
    # test_dir = os.path.join(ds_dir, 'test')

    test_nr_ds = DNSDataset_test_no_reverb(nr_dir, cut_len)
    test_wr_ds = DNSDataset_test_no_reverb(wr_dir, cut_len)

    # train_dataset = torch.utils.data.DataLoader(dataset=train_ds, batch_size=batch_size, pin_memory=True, shuffle=False,
    #                                             sampler=DistributedSampler(train_ds),
    #                                             drop_last=True, num_workers=n_cpu)
    # test_dataset = torch.utils.data.DataLoader(dataset=test_ds, batch_size=batch_size, pin_memory=True, shuffle=False,
    #                                             sampler=DistributedSampler(test_ds),
    #                                            drop_last=False, num_workers=n_cpu)

    test_nr_dataset = torch.utils.data.DataLoader(dataset=test_nr_ds, batch_size=batch_size, shuffle=False,
                                               drop_last=False, num_workers=n_cpu)
    test_wr_dataset = torch.utils.data.DataLoader(dataset=test_wr_ds, batch_size=batch_size, shuffle=False,
                                               drop_last=False, num_workers=n_cpu)




    if blind_dir != None:
        test_blind_ds = DNSDataset_test_no_reverb(blind_dir, cut_len)
        test_blind_dataset = torch.utils.data.DataLoader(dataset=test_blind_ds, batch_size=batch_size, shuffle=False,
                                                         drop_last=False, num_workers=n_cpu)
        return test_nr_dataset, test_wr_dataset, test_blind_dataset
    else:return test_nr_dataset, test_wr_dataset
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir,noisy_dir, cut_len=16000*2):
        self.cut_len = cut_len
        self.clean_dir = os.path.join(data_dir, 'train')
        self.noisy_dir = os.path.join(noisy_dir, 'train_noisy')
        self.clean_wav_name = [f for f in os.listdir(self.clean_dir) if f.endswith('.wav')]
        self.clean_wav_name = natsorted(self.clean_wav_name)
        # self.clean_wav_name = os.listdir(self.clean_dir)
        # self.clean_wav_name = natsorted(self.clean_wav_name)

    def __len__(self):
        return len(self.clean_wav_name)

    def __getitem__(self, idx):
        clean_file = os.path.join(self.clean_dir, self.clean_wav_name[idx])
        noisy_file = os.path.join(self.noisy_dir, self.clean_wav_name[idx])

        clean_ds, _ = torchaudio.load(clean_file)
        noisy_ds, _ = torchaudio.load(noisy_file)
        clean_ds = clean_ds.squeeze()
        noisy_ds = noisy_ds.squeeze()
        length = len(clean_ds)
        assert length == len(noisy_ds)
        if length < self.cut_len:
            units = self.cut_len // length
            clean_ds_final = []
            noisy_ds_final = []
            for i in range(units):
                clean_ds_final.append(clean_ds)
                noisy_ds_final.append(noisy_ds)
            clean_ds_final.append(clean_ds[: self.cut_len%length])
            noisy_ds_final.append(noisy_ds[: self.cut_len%length])
            clean_ds = torch.cat(clean_ds_final, dim=-1)
            noisy_ds = torch.cat(noisy_ds_final, dim=-1)
        else:
            # randomly cut 2 seconds segment
            wav_start = random.randint(0, length - self.cut_len)
            noisy_ds = noisy_ds[wav_start:wav_start + self.cut_len]
            clean_ds = clean_ds[wav_start:wav_start + self.cut_len]
            clean_ds = torch.stft(clean_ds, 400, 100, window=torch.hamming_window(400),onesided=True)
            clean_ds = torch.istft(clean_ds, 400, 100, window=torch.hamming_window(400),onesided=True)

        return clean_ds, noisy_ds, length
class DevDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir,noisy_dir, cut_len=16000*2):
        self.cut_len = cut_len
        self.clean_dir = os.path.join(data_dir, 'dev')
        self.noisy_dir = os.path.join(noisy_dir, 'dev_noisy')
        self.clean_wav_name = [f for f in os.listdir(self.clean_dir) if f.endswith('.wav')]
        self.clean_wav_name = natsorted(self.clean_wav_name)
        # self.clean_wav_name = os.listdir(self.clean_dir)
        # self.clean_wav_name = natsorted(self.clean_wav_name)

    def __len__(self):
        return len(self.clean_wav_name)

    def __getitem__(self, idx):
        clean_file = os.path.join(self.clean_dir, self.clean_wav_name[idx])
        noisy_file = os.path.join(self.noisy_dir, self.clean_wav_name[idx])

        clean_ds, _ = torchaudio.load(clean_file)
        noisy_ds, _ = torchaudio.load(noisy_file)
        clean_ds = clean_ds.squeeze()
        noisy_ds = noisy_ds.squeeze()
        length = len(clean_ds)
        assert length == len(noisy_ds)
        if length < self.cut_len:
            units = self.cut_len // length
            clean_ds_final = []
            noisy_ds_final = []
            for i in range(units):
                clean_ds_final.append(clean_ds)
                noisy_ds_final.append(noisy_ds)
            clean_ds_final.append(clean_ds[: self.cut_len%length])
            noisy_ds_final.append(noisy_ds[: self.cut_len%length])
            clean_ds = torch.cat(clean_ds_final, dim=-1)
            noisy_ds = torch.cat(noisy_ds_final, dim=-1)
        else:
            # randomly cut 2 seconds segment
            wav_start = random.randint(0, length - self.cut_len)
            noisy_ds = noisy_ds[wav_start:wav_start + self.cut_len]
            clean_ds = clean_ds[wav_start:wav_start + self.cut_len]
            clean_ds = torch.stft(clean_ds, 400, 100, window=torch.hamming_window(400),onesided=True)
            clean_ds = torch.istft(clean_ds, 400, 100, window=torch.hamming_window(400),onesided=True)

        return clean_ds, noisy_ds, length

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir,noisy_dir, cut_len=16000*2):
        self.cut_len = cut_len
        self.clean_dir = os.path.join(data_dir, 'test')
        self.noisy_dir = os.path.join(noisy_dir, 'test_noisy')
        self.clean_wav_name = [f for f in os.listdir(self.clean_dir) if f.endswith('.wav')]
        self.clean_wav_name = natsorted(self.clean_wav_name)
        # self.clean_wav_name = os.listdir(self.clean_dir)
        # self.clean_wav_name = natsorted(self.clean_wav_name)

    def __len__(self):
        return len(self.clean_wav_name)

    def __getitem__(self, idx):
        clean_file = os.path.join(self.clean_dir, self.clean_wav_name[idx])
        noisy_file = os.path.join(self.noisy_dir, self.clean_wav_name[idx])

        clean_ds, _ = torchaudio.load(clean_file)
        noisy_ds, _ = torchaudio.load(noisy_file)
        clean_ds = clean_ds.squeeze()
        noisy_ds = noisy_ds.squeeze()
        length = len(clean_ds)
        assert length == len(noisy_ds)
        if length < self.cut_len:
            units = self.cut_len // length
            clean_ds_final = []
            noisy_ds_final = []
            for i in range(units):
                clean_ds_final.append(clean_ds)
                noisy_ds_final.append(noisy_ds)
            clean_ds_final.append(clean_ds[: self.cut_len%length])
            noisy_ds_final.append(noisy_ds[: self.cut_len%length])
            clean_ds = torch.cat(clean_ds_final, dim=-1)
            noisy_ds = torch.cat(noisy_ds_final, dim=-1)
        else:
            # randomly cut 2 seconds segment
            wav_start = random.randint(0, length - self.cut_len)
            noisy_ds = noisy_ds[wav_start:wav_start + self.cut_len]
            clean_ds = clean_ds[wav_start:wav_start + self.cut_len]
            clean_ds = torch.stft(clean_ds, 400, 100, window=torch.hamming_window(400),onesided=True)
            clean_ds = torch.istft(clean_ds, 400, 100, window=torch.hamming_window(400),onesided=True)

        return clean_ds, noisy_ds, length
# class U_Dataset(torch.utils.data.Dataset):
#     def __init__(self, data_dir,noisy_dir, cut_len=16000*2):
#         self.cut_len = cut_len
#         self.clean_dir = os.path.join(data_dir, 'train')
#         self.noisy_dir = os.path.join(noisy_dir, 'train_noisy')
#         self.clean_wav_name = [f for f in os.listdir(self.clean_dir) if f.endswith('.wav')]
#         self.clean_wav_name = natsorted(self.clean_wav_name)
#         # self.clean_wav_name = os.listdir(self.clean_dir)
#         # self.clean_wav_name = natsorted(self.clean_wav_name)
#
#     def __len__(self):
#         return len(self.clean_wav_name)
#
#     def __getitem__(self, idx):
#         clean_file = os.path.join(self.clean_dir, self.clean_wav_name[idx])
#         noisy_file = os.path.join(self.noisy_dir, self.clean_wav_name[idx])
#
#         clean_ds, _ = torchaudio.load(clean_file)
#         noisy_ds, _ = torchaudio.load(noisy_file)
#         clean_ds = clean_ds.squeeze()
#         noisy_ds = noisy_ds.squeeze()
#         length = len(clean_ds)
#         assert length == len(noisy_ds)
#         if length < self.cut_len:
#             units = self.cut_len // length
#             clean_ds_final = []
#             noisy_ds_final = []
#             for i in range(units):
#                 clean_ds_final.append(clean_ds)
#                 noisy_ds_final.append(noisy_ds)
#             clean_ds_final.append(clean_ds[: self.cut_len%length])
#             noisy_ds_final.append(noisy_ds[: self.cut_len%length])
#             clean_ds = torch.cat(clean_ds_final, dim=-1)
#             noisy_ds = torch.cat(noisy_ds_final, dim=-1)
#         else:
#             # randomly cut 2 seconds segment
#             wav_start = random.randint(0, length - self.cut_len)
#             noisy_ds = noisy_ds[wav_start:wav_start + self.cut_len]
#             clean_ds = clean_ds[wav_start:wav_start + self.cut_len]
#             clean_ds = torch.stft(clean_ds, 400, 100, window=torch.hamming_window(400),onesided=True)
#             clean_ds = torch.istft(clean_ds, 400, 100, window=torch.hamming_window(400),onesided=True)
#
#         return clean_ds, noisy_ds, length
# class U_DevDataset(torch.utils.data.Dataset):
#     def __init__(self, data_dir,noisy_dir, cut_len=16000*2):
#         self.cut_len = cut_len
#         self.clean_dir = os.path.join(data_dir, 'dev')
#         self.noisy_dir = os.path.join(noisy_dir, 'dev_noisy')
#         self.clean_wav_name = [f for f in os.listdir(self.clean_dir) if f.endswith('.wav')]
#         self.clean_wav_name = natsorted(self.clean_wav_name)
#         # self.clean_wav_name = os.listdir(self.clean_dir)
#         # self.clean_wav_name = natsorted(self.clean_wav_name)
#
#     def __len__(self):
#         return len(self.clean_wav_name)
#
#     def __getitem__(self, idx):
#         clean_file = os.path.join(self.clean_dir, self.clean_wav_name[idx])
#         noisy_file = os.path.join(self.noisy_dir, self.clean_wav_name[idx])
#
#         clean_ds, _ = torchaudio.load(clean_file)
#         noisy_ds, _ = torchaudio.load(noisy_file)
#         clean_ds = clean_ds.squeeze()
#         noisy_ds = noisy_ds.squeeze()
#         length = len(clean_ds)
#         assert length == len(noisy_ds)
#         if length < self.cut_len:
#             units = self.cut_len // length
#             clean_ds_final = []
#             noisy_ds_final = []
#             for i in range(units):
#                 clean_ds_final.append(clean_ds)
#                 noisy_ds_final.append(noisy_ds)
#             clean_ds_final.append(clean_ds[: self.cut_len%length])
#             noisy_ds_final.append(noisy_ds[: self.cut_len%length])
#             clean_ds = torch.cat(clean_ds_final, dim=-1)
#             noisy_ds = torch.cat(noisy_ds_final, dim=-1)
#         else:
#             # randomly cut 2 seconds segment
#             wav_start = random.randint(0, length - self.cut_len)
#             noisy_ds = noisy_ds[wav_start:wav_start + self.cut_len]
#             clean_ds = clean_ds[wav_start:wav_start + self.cut_len]
#             clean_ds = torch.stft(clean_ds, 400, 100, window=torch.hamming_window(400),onesided=True)
#             clean_ds = torch.istft(clean_ds, 400, 100, window=torch.hamming_window(400),onesided=True)
#
#         return clean_ds, noisy_ds, length
#
# class U_TestDataset(torch.utils.data.Dataset):
#     def __init__(self, data_dir,noisy_dir, cut_len=16000*2):
#         self.cut_len = cut_len
#         self.clean_dir = os.path.join(data_dir, 'test')
#         self.noisy_dir = os.path.join(noisy_dir, 'test_noisy')
#         self.clean_wav_name = [f for f in os.listdir(self.clean_dir) if f.endswith('.wav')]
#         self.clean_wav_name = natsorted(self.clean_wav_name)
#         # self.clean_wav_name = os.listdir(self.clean_dir)
#         # self.clean_wav_name = natsorted(self.clean_wav_name)
#
#     def __len__(self):
#         return len(self.clean_wav_name)
#
#     def __getitem__(self, idx):
#         clean_file = os.path.join(self.clean_dir, self.clean_wav_name[idx])
#         noisy_file = os.path.join(self.noisy_dir, self.clean_wav_name[idx])
#
#         clean_ds, _ = torchaudio.load(clean_file)
#         noisy_ds, _ = torchaudio.load(noisy_file)
#         clean_ds = clean_ds.squeeze()
#         noisy_ds = noisy_ds.squeeze()
#         length = len(clean_ds)
#         assert length == len(noisy_ds)
#         if length < self.cut_len:
#             units = self.cut_len // length
#             clean_ds_final = []
#             noisy_ds_final = []
#             for i in range(units):
#                 clean_ds_final.append(clean_ds)
#                 noisy_ds_final.append(noisy_ds)
#             clean_ds_final.append(clean_ds[: self.cut_len%length])
#             noisy_ds_final.append(noisy_ds[: self.cut_len%length])
#             clean_ds = torch.cat(clean_ds_final, dim=-1)
#             noisy_ds = torch.cat(noisy_ds_final, dim=-1)
#         else:
#             # randomly cut 2 seconds segment
#             wav_start = random.randint(0, length - self.cut_len)
#             noisy_ds = noisy_ds[wav_start:wav_start + self.cut_len]
#             clean_ds = clean_ds[wav_start:wav_start + self.cut_len]
#             clean_ds = torch.stft(clean_ds, 400, 100, window=torch.hamming_window(400),onesided=True)
#             clean_ds = torch.istft(clean_ds, 400, 100, window=torch.hamming_window(400),onesided=True)
#
#         return clean_ds, noisy_ds, length
class THCHS30DDevSet(torch.utils.data.Dataset):
    def __init__(self, data_dir, cut_len=16000 * 2):
        self.cut_len = cut_len
        # self.data_dir = data_dir
        self.clean_dir = os.path.join(data_dir, 'data')
        self.noisy_dir = os.path.join(data_dir, 'dev')
        self.noisy_files = []
        self.clean_files = []
        # Load training noise and clean files

        self.load_files(self.noisy_dir)

    def load_files(self, noisy_dir):
        # 找到所有的 .wav 文件和 .trn 文件
        for file in os.listdir(noisy_dir):
            if file.endswith('.wav'):
                wav_file_path = os.path.join(noisy_dir, file)
                self.noisy_files.append(wav_file_path)
                wav_file = wav_file_path.strip()[39:]
                self.clean_files.append(wav_file)

    # def parse_trn_line(self, line):
    #     # 假设每行包含干净音频的完整路径，去除换行符和空白字符
    #     return line.strip()[8:-4]  # 去掉前八个字符和最后四个字符

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_file = os.path.join(self.clean_dir, self.clean_files[idx])

        noisy_file = os.path.join(self.noisy_dir, self.noisy_files[idx])

        clean_ds, _ = torchaudio.load(clean_file)
        noisy_ds, _ = torchaudio.load(noisy_file)
        clean_ds = clean_ds.squeeze()
        noisy_ds = noisy_ds.squeeze()

        length = len(clean_ds)
        # n_length = len(noisy_ds)
        # print(length, n_length)
        assert length == len(noisy_ds)
        if length < self.cut_len:
            units = self.cut_len // length
            clean_ds_final = []
            noisy_ds_final = []
            for i in range(units):
                clean_ds_final.append(clean_ds)
                noisy_ds_final.append(noisy_ds)
            clean_ds_final.append(clean_ds[: self.cut_len%length])
            noisy_ds_final.append(noisy_ds[: self.cut_len%length])
            clean_ds = torch.cat(clean_ds_final, dim=-1)
            noisy_ds = torch.cat(noisy_ds_final, dim=-1)
        else:
            # randomly cut 2 seconds segment
            wav_start = random.randint(0, length - self.cut_len)
            noisy_ds = noisy_ds[wav_start:wav_start + self.cut_len]
            clean_ds = clean_ds[wav_start:wav_start + self.cut_len]
            clean_ds = torch.stft(clean_ds, 400, 100, window=torch.hamming_window(400),onesided=True)
            clean_ds = torch.istft(clean_ds, 400, 100, window=torch.hamming_window(400),onesided=True)

        return clean_ds, noisy_ds, length
def load_base_data(data_dir,noisy_dir, batch_size, n_cpu, cut_len):
    torchaudio.set_audio_backend("sox_io")         # in linux
    # train_dir = os.path.join(ds_dir, 'train')
    # test_dir = os.path.join(ds_dir, 'test')
    # train_ds = THCHS30Dataset(data_dir, cut_len)
    # test_ds = THCHS30DDevSet(data_dir, cut_len)
    train_ds = TrainDataset(data_dir,noisy_dir, cut_len)
    test_ds = TestDataset(data_dir,noisy_dir, cut_len)
    train_dataset = torch.utils.data.DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True,
                                                drop_last=True, num_workers=n_cpu, persistent_workers=True,
                                                pin_memory=True)
    test_dataset = torch.utils.data.DataLoader(dataset=test_ds, batch_size=1, shuffle=False,
                                               drop_last=False, num_workers=n_cpu, persistent_workers=True,
                                               pin_memory=True)

    # train_dataset = DataLoaderX(dataset=train_ds, batch_size=batch_size, shuffle=True, pin_memory=True,
    #                             drop_last=True, num_workers=n_cpu)
    # test_dataset = DataLoaderX(dataset=test_ds, batch_size=batch_size, shuffle=False, pin_memory=True,
    #                            drop_last=False, num_workers=n_cpu)

    return train_dataset, test_dataset
def load_ft_data(data_dir,noisy_dir, batch_size, n_cpu, cut_len):
    torchaudio.set_audio_backend("sox_io")         # in linux
    # train_dir = os.path.join(ds_dir, 'train')
    # test_dir = os.path.join(ds_dir, 'test')
    # train_ds = THCHS30Dataset(data_dir, cut_len)
    # test_ds = THCHS30DDevSet(data_dir, cut_len)
    dev_ds = DevDataset(data_dir,noisy_dir, cut_len)
    test_ds = TestDataset(data_dir,noisy_dir, cut_len)
    dev_dataset = torch.utils.data.DataLoader(dataset=dev_ds, batch_size=batch_size, shuffle=True,
                                                drop_last=True, num_workers=n_cpu, persistent_workers=True,
                                                pin_memory=True)
    test_dataset = torch.utils.data.DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False,
                                               drop_last=False, num_workers=n_cpu, persistent_workers=True,
                                               pin_memory=True)

    # train_dataset = DataLoaderX(dataset=train_ds, batch_size=batch_size, shuffle=True, pin_memory=True,
    #                             drop_last=True, num_workers=n_cpu)
    # test_dataset = DataLoaderX(dataset=test_ds, batch_size=batch_size, shuffle=False, pin_memory=True,
    #                            drop_last=False, num_workers=n_cpu)

    return dev_dataset, test_dataset
def load_data(ds_dir, batch_size, n_cpu, cut_len):
    torchaudio.set_audio_backend("sox_io")         # in linux
    # train_dir = os.path.join(ds_dir, 'train')
    # test_dir = os.path.join(ds_dir, 'test')

    train_ds = DemandDataset(ds_dir, cut_len)
    test_ds = DemandDataset_1(ds_dir, cut_len)

    # train_dataset = torch.utils.data.DataLoader(dataset=train_ds, batch_size=batch_size, pin_memory=True, shuffle=False,
    #                                             sampler=DistributedSampler(train_ds),
    #                                             drop_last=True, num_workers=n_cpu)
    # test_dataset = torch.utils.data.DataLoader(dataset=test_ds, batch_size=batch_size, pin_memory=True, shuffle=False,
    #                                             sampler=DistributedSampler(test_ds),
    #                                            drop_last=False, num_workers=n_cpu)

    train_dataset = torch.utils.data.DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True,
                                                drop_last=True, num_workers=n_cpu,persistent_workers=True,pin_memory=True)
    test_dataset = torch.utils.data.DataLoader(dataset=test_ds, batch_size=1, shuffle=True,
                                               drop_last=False, num_workers=1,persistent_workers=True,pin_memory=True)



    # train_dataset = DataLoaderX(dataset=train_ds, batch_size=batch_size, shuffle=True, pin_memory=True,
    #                             drop_last=True, num_workers=n_cpu)
    # test_dataset = DataLoaderX(dataset=test_ds, batch_size=batch_size, shuffle=False, pin_memory=True,
    #                            drop_last=False, num_workers=n_cpu)

    return train_dataset, test_dataset
class DevDataset_E(torch.utils.data.Dataset):
    def __init__(self, noisy_dir, cut_len=16000*2):
        self.cut_len = cut_len
        self.clean_dir = os.path.join(noisy_dir, 'train')
        self.noisy_dir = os.path.join(noisy_dir, 'train_noisy')
        self.clean_wav_name = [f for f in os.listdir(self.clean_dir) if f.endswith('.wav')]
        self.clean_wav_name = natsorted(self.clean_wav_name)
        # self.clean_wav_name = os.listdir(self.clean_dir)
        # self.clean_wav_name = natsorted(self.clean_wav_name)

    def __len__(self):
        return len(self.clean_wav_name)

    def __getitem__(self, idx):
        clean_file = os.path.join(self.clean_dir, self.clean_wav_name[idx])
        noisy_file = os.path.join(self.noisy_dir, self.clean_wav_name[idx])

        clean_ds, _ = torchaudio.load(clean_file)
        noisy_ds, _ = torchaudio.load(noisy_file)
        clean_ds = clean_ds.squeeze()
        noisy_ds = noisy_ds.squeeze()
        length = len(clean_ds)
        assert length == len(noisy_ds)
        if length < self.cut_len:
            units = self.cut_len // length
            clean_ds_final = []
            noisy_ds_final = []
            for i in range(units):
                clean_ds_final.append(clean_ds)
                noisy_ds_final.append(noisy_ds)
            clean_ds_final.append(clean_ds[: self.cut_len%length])
            noisy_ds_final.append(noisy_ds[: self.cut_len%length])
            clean_ds = torch.cat(clean_ds_final, dim=-1)
            noisy_ds = torch.cat(noisy_ds_final, dim=-1)
        else:
            # randomly cut 2 seconds segment
            wav_start = random.randint(0, length - self.cut_len)
            noisy_ds = noisy_ds[wav_start:wav_start + self.cut_len]
            clean_ds = clean_ds[wav_start:wav_start + self.cut_len]
            clean_ds = torch.stft(clean_ds, 400, 100, window=torch.hamming_window(400),onesided=True)
            clean_ds = torch.istft(clean_ds, 400, 100, window=torch.hamming_window(400),onesided=True)

        return clean_ds, noisy_ds, length

def load_ft_E_data(data_dir,noisy_dir, batch_size, n_cpu, cut_len):
    torchaudio.set_audio_backend("sox_io")         # in linux
    # train_dir = os.path.join(ds_dir, 'train')
    # test_dir = os.path.join(ds_dir, 'test')

    train_ds = DevDataset_E(noisy_dir, cut_len)
    test_ds = DemandDataset_1(data_dir, cut_len)

    # train_dataset = torch.utils.data.DataLoader(dataset=train_ds, batch_size=batch_size, pin_memory=True, shuffle=False,
    #                                             sampler=DistributedSampler(train_ds),
    #                                             drop_last=True, num_workers=n_cpu)
    # test_dataset = torch.utils.data.DataLoader(dataset=test_ds, batch_size=batch_size, pin_memory=True, shuffle=False,
    #                                             sampler=DistributedSampler(test_ds),
    #                                            drop_last=False, num_workers=n_cpu)

    train_dataset = torch.utils.data.DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True,
                                                drop_last=True, num_workers=n_cpu,persistent_workers=True,pin_memory=True)
    test_dataset = torch.utils.data.DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False,
                                               drop_last=False, num_workers=n_cpu,persistent_workers=True,pin_memory=True)



    # train_dataset = DataLoaderX(dataset=train_ds, batch_size=batch_size, shuffle=True, pin_memory=True,
    #                             drop_last=True, num_workers=n_cpu)
    # test_dataset = DataLoaderX(dataset=test_ds, batch_size=batch_size, shuffle=False, pin_memory=True,
    #                            drop_last=False, num_workers=n_cpu)

    return train_dataset, test_dataset
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())