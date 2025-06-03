import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
# from geomloss import SamplesLoss
# from fairseq.models.wav2vec import Wav2VecModel
def Loss_si_snr(est,clean):
    s_t = (est * clean)/(clean**2)
    loss =  -torch.log10((torch.norm(s_t)**2)/(torch.norm(est-s_t)**2+1e-8) +1e-8)
    return loss
class LearnableSigmoid(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)
class LearnableSigmoid_2d(nn.Module):
    def __init__(self, in_features, beta=2):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)
def kaiming_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def get_padding_2d(kernel_size, dilation=(1, 1)):
    return (int((kernel_size[0]*dilation[0] - dilation[0])/2), int((kernel_size[1]*dilation[1] - dilation[1])/2))



def ri_stft(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    hann_window = torch.hann_window(win_size).to(y.device)
    stft_spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                           center=center, pad_mode='reflect', normalized=False, return_complex=True)
    stft_spec = torch.view_as_real(stft_spec)
    mag = torch.sqrt(stft_spec.pow(2).sum(-1) + (1e-9))
    pha = torch.atan2(stft_spec[:, :, :, 1] + (1e-10), stft_spec[:, :, :, 0] + (1e-5))
    # Magnitude Compression
    mag = torch.pow(mag, compress_factor)
    r,i = mag * torch.cos(pha), mag * torch.sin(pha)

    return r.unsqueeze(1),i.unsqueeze(1)
def power_compress_pha(x):
    real = x[..., 0]
    imag = x[..., 1]
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag**0.3
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], 1),phase
def power_compress(x):           # origin
    real = x[..., 0]
    imag = x[..., 1]
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag**0.3
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], 1)



def power_uncompress(real, imag):


    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag**(1./0.3)
    # mag = mag**(1./nn.Sigmoid())
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], -1)
def pcs(x):           # origin
    PCS400 = torch.ones(201).to('cuda')
    PCS400[0:3] = 1
    PCS400[3:5] = 1.070175439
    PCS400[5:8] = 1.182456140
    PCS400[8:10] = 1.287719298
    PCS400[10:110] = 1.4  # Pre Set
    PCS400[110:130] = 1.322807018
    PCS400[130:160] = 1.238596491
    PCS400[160:190] = 1.161403509
    PCS400[190:202] = 1.077192982
    real = x[:, 0,:,:]
    imag = x[:, 1,:,:]
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = (PCS400 * torch.log1p(mag.permute(0,2,1))).permute(0,2,1)
    # mag = torch.transpose(Lp, (1, 0))
    # mag = mag**0.3
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], 1)

def mag_pcs(x):           # origin
    PCS = torch.ones(257).to('cuda')
    PCS[0:3] = 1
    PCS[3:6] = 1.070175439
    PCS[6:9] = 1.182456140
    PCS[9:12] = 1.287719298
    PCS[12:138] = 1.4  # Pre Set
    PCS[138:166] = 1.322807018
    PCS[166:200] = 1.238596491
    PCS[200:241] = 1.161403509
    PCS[241:256] = 1.077192982
    real = x[..., 0]
    imag = x[..., 1]
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = (PCS * torch.log1p(mag.permute(0,2,1))).permute(0,2,1)
    # mag = torch.transpose(Lp, (1, 0))
    # mag = mag**0.3
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], 1),phase

def mag_ipcs(real, imag):


    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = torch.expm1(mag)
    # mag = mag**(1./nn.Sigmoid())
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    rec = phase*torch.exp(1j*phase)
    return real_compress, imag_compress,rec
def ipcs(real, imag):


    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = torch.expm1(mag)
    # mag = mag**(1./nn.Sigmoid())
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    rec = phase*torch.exp(1j*phase)
    return torch.stack([real_compress, imag_compress], -1),rec
def power_uncompress_pha(real, imag):


    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag**(1./0.3)
    # mag = mag**(1./nn.Sigmoid())
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], -1),phase
def pcs_uncompress(real, imag):
    PCS400 = torch.ones(201).to('cuda')
    PCS400[0:3] = 1
    PCS400[3:5] = 1.070175439
    PCS400[5:8] = 1.182456140
    PCS400[8:10] = 1.287719298
    PCS400[10:110] = 1.4  # Pre Set
    PCS400[110:130] = 1.322807018
    PCS400[130:160] = 1.238596491
    PCS400[160:190] = 1.161403509
    PCS400[190:202] = 1.077192982

    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = (PCS400 * torch.log1p(mag.permute(0,1,3,2))).permute(0,1,3,2)
    # mag = mag**(1./nn.Sigmoid())
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return real_compress, imag_compress
def lsd_loss(clean, estimate):
    alpha = 0.5
    lsd_m =  torch.log(torch.mean(torch.norm(clean.pow(alpha)-estimate.pow(alpha), dim=-1, keepdim=True)**2+1e-8))
    return lsd_m
def snr_loss(clean, estimate):
    snr = - torch.mean(torch.log10(torch.abs(clean) ** 2 / torch.abs((estimate - clean) ** 2 + 1e-8) + 1e-8))
    return snr
# def PerceptualLoss(model,est,clean):
#     y_hat, y = map(model, [est, clean])
#     return SamplesLoss(y_hat, y)
# class PerceptualLoss(nn.Module):
#     def __init__(self, model_type='wav2vec', PRETRAINED_MODEL_PATH = '/home/exp/cmgan/src/pt-models/wav2vec_big_960h.pt'):
#         super().__init__()
#         self.model_type = model_type
#         self.wass_dist = SamplesLoss()
#         # if model_type == 'wav2vec':
#
#         ckpt = torch.load(PRETRAINED_MODEL_PATH)
#         self.model = Wav2VecModel.build_model(ckpt['args'], task=None)
#         self.model.load_state_dict(ckpt['model'])
#         self.model = self.model.feature_extractor
#         self.model.eval()
#         # else:
#         #     print('Please assign a loss model')
#         #     sys.exit()
#
#     def forward(self, y_hat, y):
#         y_hat, y = map(self.model, [y_hat, y])
#         return self.wass_dist(y_hat, y)
        # for PFPL-W-MAE or PFPL-W
        # return torch.abs(y_hat - y).mean()
def get_augmented_input(x, t):
    padded = []

    # 添加过去/未来帧
    for i in range(-2, 2):
        # 不同于连续的 t - m 和 t + n
        idx = t + i
        if 0 <= idx < x.size(1):  # 确保索引在边界内
            padded.append(x[:, idx:idx + 1, :])  # 维度应保持一致

    return torch.cat(padded, dim=1)  # 在时间维度上拼接
