# from thop import profile
import logging
import argparse
import logging
import os
import time
from torch.utils.tensorboard import SummaryWriter
import json
import numpy as np
import torch.distributed as dist
import torch
import torch.nn.functional as F
from loguru import logger
from torchinfo import summary
from data import dataloader
from models import discriminator
from models.discriminator import *
from models.generator_td import  TSCNet
from utils import *

def compute_dgrad_angle(grad1,grad2):
    dot_product = torch.dot(grad1, grad2)
    norm_clean_grad = torch.norm(grad1)
    norm_est_grad = torch.norm(grad2)
    cos_theta = dot_product / (norm_clean_grad * norm_est_grad)
    theta = torch.acos(cos_theta)
    angle_degrees = torch.rad2deg(theta)
    return angle_degrees
def phase_losses(phase_r, phase_g):

    dim_freq = 400// 2 + 1
    dim_time = phase_r.size(-1)
    # print(phase_g.shape)
    # print(phase_r.shape)
    gd_matrix = (torch.triu(torch.ones(dim_freq, dim_freq), diagonal=1) - torch.triu(torch.ones(dim_freq, dim_freq), diagonal=2) - torch.eye(dim_freq)).to(phase_g.device)
    gd_r = torch.matmul(phase_r.permute(0, 2, 1), gd_matrix)    #沿频率轴求导，群时延
    gd_g = torch.matmul(phase_g.permute(0, 2, 1), gd_matrix)

    iaf_matrix = (torch.triu(torch.ones(dim_time, dim_time), diagonal=1) - torch.triu(torch.ones(dim_time, dim_time), diagonal=2) - torch.eye(dim_time)).to(phase_g.device)
    iaf_r = torch.matmul(phase_r, iaf_matrix)                   #沿时间轴求导，瞬时角频率
    iaf_g = torch.matmul(phase_g, iaf_matrix)

    ip_loss = torch.mean(anti_wrapping_function(phase_r-phase_g))
    gd_loss = torch.mean(anti_wrapping_function(gd_r-gd_g))
    iaf_loss = torch.mean(anti_wrapping_function(iaf_r-iaf_g))

    return ip_loss, gd_loss, iaf_loss
def anti_wrapping_function(x):

    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)





def compute_dgrad(gc,ge,gn):
    clean_grad = gc
    est_grad = ge
    if gn is not None:
        n_grad = gn
    else:
        n_grad = None
    angle_ce = compute_dgrad_angle(clean_grad, est_grad)
    if angle_ce <= 90:
        w1, w2 = 1, 1
        if n_grad is not None and compute_dgrad_angle(w1 * clean_grad + w2 * est_grad, n_grad) <= 90:
            w3 = 1
        else:
            dot_product_cn = torch.dot(clean_grad, n_grad)
            dot_product_en = torch.dot(est_grad, n_grad)
            norm_n_grad = torch.norm(n_grad)
            w3 = -dot_product_cn / norm_n_grad - dot_product_en / norm_n_grad
    else:
        w1 = 1
        w2 = -torch.dot(clean_grad, est_grad) / torch.norm(n_grad)
        if n_grad is not None and compute_dgrad_angle(w1 * clean_grad + w2 * est_grad, n_grad) <= 90:
            w3 = 1
        else:
            dot_product_cn = torch.dot(clean_grad, n_grad)
            dot_product_en = torch.dot(est_grad, n_grad)
            dot_product_ce = torch.dot(clean_grad, est_grad)
            norm_n_grad = torch.norm(n_grad)
            norm_est_grad = torch.norm(est_grad)
            w3 = -dot_product_cn / norm_n_grad + dot_product_ce * dot_product_en / (norm_est_grad * norm_n_grad)
    return w1,w2,w3

logger.add('20231008.log',filter='debug')
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--save_epoch", type=int, default=25, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--log_interval", type=int, default=10)
parser.add_argument("--decay_epoch", type=int, default=10, help="epoch from which to start lr decay")
parser.add_argument("--init_lr", type=float, default=5e-4, help="initial learning rate")
parser.add_argument("--cut_len", type=int, default=16000*2, help="cut length, default is 2 seconds in denoise "
                                                                 "and dereverberation")
parser.add_argument("--data_dir", type=str, default='/home/dataset/Voicebank/noisy-vctk-16k',
                    help="dir of VCTK-DEMAND dataset")
# parser.add_argument("--data_dir", type=str, default='/home/xyj/datasets/uyghur/',
#                     help="dir of dataset")
# parser.add_argument("--noisy_dir", type=str, default='/home/xyj/datasets/uyghur/',
#                     help="dir of noisy data")
parser.add_argument("--save_model_dir", type=str, default='/home/xyj/Experiment/CMG-v1/src/TdNet_Vb_100epochs/',
                    help="dir of saved model")
parser.add_argument("--loss_weights", type=list, default=[0.1, 0.9,0.2,0.2,0.05],
                    help="weights of RI components, magnitude, time loss, and Metric Disc")
args = parser.parse_args()

logging.basicConfig(format='%(message)s', level=logging.INFO)
class Trainer:
    def __init__(self, train_ds, test_ds,):

        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.n_fft = 400
        self.hop = 100
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.model = TSCNet(num_channel=64, num_features=self.n_fft // 2 + 1).to(self.device)
        summary(self.model, [(1, 2, args.cut_len//self.hop+1, int(self.n_fft/2)+1)])

        self.discriminator = discriminator.Discriminator(ndf=16).to(self.device)
        summary(self.discriminator, [(1, 1, int(self.n_fft/2)+1, args.cut_len//self.hop+1),
                                     (1, 1, int(self.n_fft/2)+1, args.cut_len//self.hop+1)])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.init_lr,betas=(0.81,0.99),weight_decay=0.00001)
        self.optimizer_disc = torch.optim.AdamW(self.discriminator.parameters(), lr=2*args.init_lr,betas=(0.8,0.999),weight_decay=0.00001)
        self.loss_history = []
        self.loss_train = []
        self.disc_loss_history = []
        self.loss_valid = []
        self.pesq_history = []
        self.args = args
        self.writer = SummaryWriter(log_dir=os.path.join(args.save_model_dir, 'logs'))
    def train_step(self, batch):

        clean = batch[0].to(self.device)
        noisy = batch[1].to(self.device)
        one_labels = torch.ones(args.batch_size).to(self.device)
        # Normalization
        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
        noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
        noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(clean * c, 0, 1)


        n = 5e-4 * torch.randn_like(noisy)
        noisy = noisy + n

        self.optimizer.zero_grad()
        noisy_spec = torch.stft(noisy, self.n_fft, self.hop, window=torch.hamming_window(self.n_fft).to(self.device),
                                onesided=True)
        clean_spec = torch.stft(clean, self.n_fft, self.hop, window=torch.hamming_window(self.n_fft).to(self.device),
                                onesided=True)

        noisy_spec, p_n = power_compress_pha(noisy_spec)[0], power_compress_pha(noisy_spec)[1]

        clean_spec, p_c = power_compress_pha(clean_spec)[0], power_compress_pha(clean_spec)[1]

        clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
        clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)
        
        est_real, est_imag, denoised_phase = self.model(noisy_spec.permute(0, 1, 3, 2))
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)

        est_spec_uncompress, p_e = power_uncompress_pha(est_real, est_imag)[0].squeeze(1), \
            power_uncompress_pha(est_real, est_imag)[1]
        est_audio = torch.istft(est_spec_uncompress, self.n_fft, self.hop,
                                window=torch.hamming_window(self.n_fft).to(self.device), onesided=True)

        time_loss = torch.mean(torch.abs(est_audio - clean))

        est_mag = torch.sqrt(est_real ** 2 + est_imag ** 2)
        clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2)


        est_real_hat, est_imag_hat = ri_stft(est_audio, self.n_fft, self.hop, self.n_fft, 0.3)
        est_mag_hat = torch.sqrt(est_real_hat ** 2 + est_imag_hat ** 2)

        loss_gc = F.mse_loss(est_real, est_real_hat) + F.mse_loss(est_imag, est_imag_hat)
        loss_gm = F.mse_loss(est_mag, est_mag_hat)
        loss_cp = args.loss_weights[0] * loss_gc + args.loss_weights[1] * loss_gm
        predict_fake_metric = self.discriminator(clean_mag, est_mag_hat)
        gen_loss_GAN = F.mse_loss(predict_fake_metric.flatten(),
                                  one_labels.float()) 
        loss_mag = F.mse_loss(est_mag, clean_mag)
        loss_ri = F.mse_loss(est_real, clean_real) + F.mse_loss(est_imag,
                                                                clean_imag)
        loss_ip, loss_gd, loss_iaf = phase_losses(p_c, denoised_phase.permute(0, 3, 2, 1).squeeze(-1))
        loss_pha = loss_ip + loss_gd + loss_iaf

        length = est_audio.size(-1)

        loss = args.loss_weights[0] * loss_ri + args.loss_weights[1] * loss_mag + args.loss_weights[2] * time_loss \
               + args.loss_weights[3] * loss_pha + args.loss_weights[4] * gen_loss_GAN + 0.05 * loss_cp

        loss.backward()

        self.optimizer.step()

        est_audio_list = list(est_audio.detach().cpu().numpy())
        clean_audio_list = list(clean.cpu().numpy()[:, :length])
        pesq_score = discriminator.b_pesq(clean_audio_list, est_audio_list,self.device)
        if pesq_score is not None:

            self.optimizer_disc.zero_grad()

            predict_enhance_metric = self.discriminator(clean_mag,est_mag_hat.detach() )
            predict_max_metric = self.discriminator(clean_mag, clean_mag)
            clean_loss = F.mse_loss(predict_max_metric.flatten(), one_labels)
            est_loss = F.mse_loss(predict_enhance_metric.flatten(), pesq_score.to(self.device))
            clean_grad = predict_max_metric[0]
            est_grad = predict_enhance_metric[0]
            angle_ce = compute_dgrad_angle(clean_grad, est_grad)
            if angle_ce <= 90:
                w1, w2 = 1, 1
            else:
                w1 = 1
                w2 = -torch.dot(clean_grad, est_grad) / torch.norm(est_grad)
            discrim_loss_metric =w1 *  clean_loss + w2 *est_loss   #+ w3 * noisy_loss
            clean_loss.backward()
            est_loss.backward() 
            self.optimizer_disc.step()
        else:
            discrim_loss_metric = torch.tensor([0.])
        return loss.item(), discrim_loss_metric.item(),  loss_mag.item(), loss_ri.item(), loss_pha.item(),time_loss.item(),loss_cp.item(), gen_loss_GAN.item()

    @torch.no_grad()
    def test_step(self, batch):

        clean = batch[0].to(self.device)
        noisy = batch[1].to(self.device)

        one_labels = torch.ones(args.batch_size).to(self.device)
        # Normalization
        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
        noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
        noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(clean * c, 0, 1)

        self.optimizer.zero_grad()
        noisy_spec = torch.stft(noisy, self.n_fft, self.hop, window=torch.hamming_window(self.n_fft).to(self.device),
                                onesided=True)
        clean_spec = torch.stft(clean, self.n_fft, self.hop, window=torch.hamming_window(self.n_fft).to(self.device),
                                onesided=True)

        noisy_spec, p_n = power_compress_pha(noisy_spec)[0], power_compress_pha(noisy_spec)[1]

        clean_spec, p_c = power_compress_pha(clean_spec)[0], power_compress_pha(clean_spec)[1]
        clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
        clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)

        est_real, est_imag, denoised_phase = self.model(noisy_spec.permute(0, 1, 3, 2))
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
        est_spec_uncompress, p_e = power_uncompress_pha(est_real, est_imag)[0].squeeze(1), \
            power_uncompress_pha(est_real, est_imag)[1]
        est_audio = torch.istft(est_spec_uncompress, self.n_fft, self.hop,
                                window=torch.hamming_window(self.n_fft).to(self.device), onesided=True)

        time_loss = torch.mean(torch.abs(est_audio - clean))


        est_mag = torch.sqrt(est_real ** 2 + est_imag ** 2)
        clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2)


        est_real_hat, est_imag_hat = ri_stft(est_audio, self.n_fft, self.hop, self.n_fft, 0.3)
        est_mag_hat = torch.sqrt(est_real_hat ** 2 + est_imag_hat ** 2)

        loss_gc = F.mse_loss(est_real, est_real_hat) + F.mse_loss(est_imag, est_imag_hat)
        loss_gm = F.mse_loss(est_mag, est_mag_hat)
        loss_cp = args.loss_weights[0] * loss_gc + args.loss_weights[1] * loss_gm
        predict_fake_metric = self.discriminator(clean_mag, est_mag_hat)

        gen_loss_GAN = 1 * F.mse_loss(predict_fake_metric.flatten(),
                                      one_labels.float())  # + 0.3*F.mse_loss(predict_noise_metric.flatten(), one_labels.float())
        loss_mag = F.mse_loss(est_mag, clean_mag)
        loss_ri = F.mse_loss(est_real, clean_real) + F.mse_loss(est_imag,
                                                                clean_imag)
        loss_ip, loss_gd, loss_iaf = phase_losses(p_c, denoised_phase.permute(0, 3, 2, 1).squeeze(-1))
        loss_pha = loss_ip + loss_gd + loss_iaf

        length = est_audio.size(-1)

        loss = args.loss_weights[0] * loss_ri + args.loss_weights[1] * loss_mag + args.loss_weights[2] * time_loss \
               + args.loss_weights[3] * loss_pha + args.loss_weights[4] * gen_loss_GAN + 0.05 * loss_cp

        self.optimizer.step()

        r_audio = torch.split(clean, 1, dim=0)
        g_audio = torch.split(est_audio, 1, dim=0)
        est_audio_list = list(est_audio.detach().cpu().numpy())
        clean_audio_list = list(clean.cpu().numpy()[:, :length])
        pesq_score = discriminator.b_pesq(clean_audio_list, est_audio_list,self.device)
        if pesq_score is not None:
            self.optimizer_disc.zero_grad()
            predict_enhance_metric = self.discriminator(clean_mag,est_mag_hat.detach() )
            predict_max_metric = self.discriminator(clean_mag, clean_mag)
            clean_loss = F.mse_loss(predict_max_metric.flatten(), one_labels)
            est_loss = F.mse_loss(predict_enhance_metric.flatten(), pesq_score.to(self.device))
            clean_grad = predict_max_metric[0]
            est_grad = predict_enhance_metric[0]
            angle_ce = compute_dgrad_angle(clean_grad, est_grad)
            if angle_ce <= 90:
                w1, w2 = 1, 1
            else:
                w1 = 1
                w2 = -torch.dot(clean_grad, est_grad) / torch.norm(est_grad)
            discrim_loss_metric =w1 *  clean_loss + w2 *est_loss   #+ w3 * noisy_loss


            self.optimizer_disc.step()


        else:
            discrim_loss_metric = torch.tensor([0.])
        return loss.item(), discrim_loss_metric.item(),r_audio,g_audio

    def test(self):
        self.model.eval()
        self.discriminator.eval()
        gen_loss_total = 0.
        disc_loss_total = 0.
        r_total = []
        g_total = []

        for idx, batch in enumerate(self.test_ds):
            step = idx + 1
            loss, disc_loss,r,g = self.test_step(batch)
            r_total += r
            g_total += g
            gen_loss_total += loss
            disc_loss_total += disc_loss
        gen_loss_avg = gen_loss_total / step
        disc_loss_avg = disc_loss_total / step
        valid_pesq = pesq_value(r_total, g_total,sr=16000)

        template = 'PESQ score: {}, Generator loss: {}, Discriminator loss: {},total_loss{}'
        logging.info(
            template.format(valid_pesq,gen_loss_avg, disc_loss_avg,gen_loss_avg+disc_loss_avg))
        return gen_loss_avg,disc_loss_avg,valid_pesq

    def train(self):
        scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.decay_epoch, gamma=0.7)
        scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_disc, step_size=args.decay_epoch, gamma=0.7)
        # scheduler_G = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        # scheduler_D = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_disc, gamma=0.99)
        min_loss = 1.
        best_epoch = None
        best_pesq = 0
        val_pesq = 0
        last_epoch = 0
        load_step = 0
        if os.path.exists(os.path.join(args.save_model_dir, f'last_save.pth')):
            checkpoint = torch.load(os.path.join(args.save_model_dir, f'last_save.pth'), map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.discriminator.load_state_dict(checkpoint['disc_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.optimizer_disc.load_state_dict(checkpoint['disc_optimizer_state_dict'])
            self.loss_history = checkpoint['loss_history']
            self.disc_loss_history = checkpoint['disc_loss_history']
            last_epoch = checkpoint['epoch']
            # load_step = checkpoint['current_step']
            # val_pesq = checkpoint['val_pesq']
        for epoch in range(max(0, last_epoch), args.epochs):

            self.model.train()
            self.discriminator.train()
            for idx, batch in enumerate(self.train_ds):
                t_start = time.time()
                if load_step != 0:
                    step = load_step
                else:
                    step = idx + 1

                loss, disc_loss, l_m, l_c, l_p, l_t, l_stft, l_Adv_g = self.train_step(batch)
                l_m, l_c, l_p, l_t, l_stft, l_Adv_g = l_m, l_c, l_p, l_t, l_stft, l_Adv_g * \
                                                                                  args.loss_weights[4]
                t_end = time.time()
                current_step = step
                total_step = len(self.train_ds) * epoch + step
                if total_step % 1000 == 0:
                    if epoch > 35 and total_step % 1000==0:
                        torch.cuda.empty_cache()
                        new_pesq = self.test()[2]
                        if new_pesq > val_pesq:
                            val_pesq = new_pesq
                            higher_pesq = new_pesq
                            torch.save(self.model.state_dict(),
                                       os.path.join(args.save_model_dir, str(higher_pesq)[:5] + 'better_model.pth'))
                    current_save_path = os.path.join(args.save_model_dir, f'last_save.pth')
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'disc_state_dict': self.discriminator.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'disc_optimizer_state_dict': self.optimizer_disc.state_dict(),
                        'loss_history': self.loss_history,
                        'disc_loss_history': self.disc_loss_history,
                        'epoch': epoch,
                        'load_step': step+1,
                        'val_pesq': val_pesq
                    }, current_save_path)

                self.loss_history.append(loss)
                self.disc_loss_history.append(disc_loss)
                spend_time = t_end - t_start
                template = 'time {}, Epoch {}, Step {}, L_G: {}, L_D: {}; L_m: {}, L_c: {}, L_p: {}, L_t: {}, L_cp: {}, L_Adv_g: {}'
                if (step % self.args.log_interval) == 0:
                    logging.info(template.format(str(spend_time)[:5], epoch, step, str(loss)[:5], str(disc_loss)[:5],
                                                 str(l_m)[:5], str(l_c)[:5], str(l_p)[:5], str(l_t)[:5],
                                                 str(l_stft)[:5], str(l_Adv_g)[:5]))
                # 记录每一项损失
                self.writer.add_scalar('Loss/l_m', l_m, step)  # 记录 l_m
                self.writer.add_scalar('Loss/l_c', l_c, step)  # 记录 l_c
                self.writer.add_scalar('Loss/l_p', l_p, step)  # 记录 l_p
                self.writer.add_scalar('Loss/l_t', l_t, step)  # 记录 l_t
                self.writer.add_scalar('Loss/l_stft', l_stft, step)  # 记录 l_stft
                self.writer.add_scalar('Loss/l_Adv_g', l_Adv_g, step)  # 记录 l_Adv_g
                self.writer.add_scalar('Loss/total', loss, step)  # 记录总损失


            loss_avg = sum(self.loss_history[current_step*epoch:current_step*(epoch+1)]) / (current_step)
            self.loss_train.append(loss_avg)
            torch.cuda.empty_cache()
            gen_loss = self.test()
            valid_loss = gen_loss[0] +gen_loss[1]
            valid_pesq =gen_loss[2]
            self.pesq_history.append(valid_pesq)
            if valid_pesq > best_pesq:
                best_pesq = valid_pesq
                maxpesq_epoch = self.model.state_dict()
            if valid_loss < min_loss:
                min_loss = valid_loss
                best_epoch = self.model.state_dict()
            path = os.path.join(args.save_model_dir,   'epoch' + str(epoch) +'-pesq:' + str(valid_pesq)[:5] + '-loss_sum:' +str(gen_loss[0]+gen_loss[1])[:6]+':' +'-g:' +str(gen_loss[0])[:5] + '-d:' + str(gen_loss[1])[:5])
            self.loss_valid.append(gen_loss[0])
            valid_loss_avg = sum(self.loss_valid) / len(self.loss_valid)
            if not os.path.exists(args.save_model_dir):
                os.makedirs(args.save_model_dir)
            torch.save(self.model.state_dict(), path)
            scheduler_G.step()
            scheduler_D.step()

            if (epoch+1) % args.save_epoch == 0:
                save_path = os.path.join(args.save_model_dir, '_' + str(min_loss) + '_min')
                torch.save(best_epoch, save_path)
                pesqmax_path = os.path.join(args.save_model_dir, '_' + 'max_pesq=' + str(best_pesq) )
                torch.save(maxpesq_epoch, pesqmax_path)
                # Save loss history
                loss_history_dir = os.path.join(args.save_model_dir, 'loss_history')
                if not os.path.exists(loss_history_dir):
                    os.makedirs(loss_history_dir)
                # Calculate average loss and disc_loss
                train_avg_loss = sum(self.loss_history) / len(self.loss_history)
                train_avg_disc_loss = sum(self.disc_loss_history) / len(self.disc_loss_history)
                train_avg_pesq = sum(self.pesq_history) / len(self.pesq_history)
                # Save loss_history
                with open(os.path.join(loss_history_dir,
                                       f'train_loss_history_epoch_{epoch}_avg_{train_avg_loss:.5f}.json'),
                          'w') as f:
                    json.dump({'loss_history': self.loss_history}, f)

                # Save disc_loss_history
                with open(os.path.join(loss_history_dir,
                                       f'train_disc_loss_history_epoch_{epoch}_avg_{train_avg_disc_loss:.5f}.json'),
                          'w') as f:
                    json.dump({'disc_loss_history': self.disc_loss_history}, f)
                # 保存 train_loss
                train_loss_path = os.path.join(loss_history_dir, f'train_loss_epoch_{epoch}_avg_{loss_avg:.5f}.json')
                with open(train_loss_path, 'w') as f:
                    json.dump({'train_loss': self.loss_train}, f)
                # 保存 valid_loss
                valid_loss_path = os.path.join(loss_history_dir, f'valid_loss_epoch_{epoch}_avg_{valid_loss_avg:.5f}.json')
                with open(valid_loss_path, 'w') as f:
                    json.dump({'valid_loss': self.loss_valid}, f)
                # 保存 valid_pesq
                valid_pesq_path = os.path.join(loss_history_dir, f'valid_pesq_epoch_{epoch}_avg_{train_avg_pesq:.5f}.json')
                with open(valid_pesq_path, 'w') as f:
                    json.dump({'valid_pesq': self.pesq_history}, f)


@logger.catch()
def main():
    cuda_device = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] =cuda_device

    torch.device('cuda:0')
    torch.autograd.set_detect_anomaly(True)
    print(args)
    available_gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print(available_gpus)
    # train_ds, test_ds = dataloader.load_base_data(args.data_dir, args.noisy_dir, args.batch_size, 12, args.cut_len)
    train_ds, test_ds = dataloader.load_data(args.data_dir, args.batch_size, 12, args.cut_len)
    trainer = Trainer(train_ds, test_ds)
    trainer.train()



if __name__ == '__main__':
    main()



