import random
from glob import glob
import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import time
import numpy as np
import importlib
import math
import logging
import torch
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torchvision import transforms
import torch.nn as nn
from Common.util import save_checkpoint, psnr, tensor2img, print_loss, show_in_board

import config as config

from Common.write_bin import read_from_bin, write_to_bin
from data.dataset_noise_mix import DataLoader_Noise_domain_noise


from model import create_model
from data.test_dataload import data_load, data_load_same_name

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.backends.cudnn.deterministic = True

logger = logging.getLogger(config.args.model)
logger.setLevel(level=logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s %(message)s')

stdhandler = logging.StreamHandler()
stdhandler.setFormatter(formatter)
logger.addHandler(stdhandler)

if not os.path.exists('./ckpt/log/'):
    os.makedirs('./ckpt/log/')

filehandler = logging.FileHandler(
    f'./ckpt/log/{config.args.model}_{config.args.label_str}_{config.args.lmbda}.log')
filehandler.setFormatter(formatter)
logger.addHandler(filehandler)

writer = SummaryWriter(f'./log/{config.args.model}/{config.args.label_str}/{str(config.args.lmbda)}')


def train_one_epoch(opt, model, train_dataloader, optimizer,
                    aux_optimizer, epoch):
    global step
    model.train()
    device = next(model.parameters()).device
    step = 0
    print_in = opt.print_num
    for i, (p_im, d_im1, d_im2) in enumerate(train_dataloader):
        p_im = p_im.to(device)
        d_im1 = d_im1.to(device)
        d_im2 = d_im2.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d_im1, d_im2)

        out_criterion = model.loss(out_net, p_im, d_im1, d_im2, opt.lmbda)

        out_criterion["loss"].backward()
        if opt.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()
        
        out_criterion["aux_loss"] = aux_loss

        if step % print_in == 0:
            print_str = f"{opt.model} {opt.label_str} {opt.lmbda} Train epoch {epoch}: [{i * len(d_im1)}/{len(train_dataloader.dataset)} ({100. * i / len(train_dataloader):.0f}%)]"
            print_l = print_loss(out_criterion)
            logger.info(print_str + print_l)

            show_in_board(
                writer,
                out_net,
                step,
                d_im1=d_im1,
                d_im2=d_im2,
                p_im=p_im)

        writer.add_scalar("Loss/Train/aux_loss", aux_loss, step)
        for kk in out_criterion:
            writer.add_scalar(
                f"Loss/Train/{out_criterion[kk]}",
                out_criterion[kk],
                step)
        step = step + 1

def train(opt):

    if opt.seed is not None:
        torch.manual_seed(opt.seed)  # fix the random values
        random.seed(opt.seed)

    train_dataloader = DataLoader_Noise_domain_noise(
        opt.train_img_dataset,
        opt.train_real_dataset,
        opt.image_size,
        opt.batch_size)

    device = "cuda" if opt.cuda and torch.cuda.is_available() else "cpu"

    net = create_model(opt)
    net = net.to(device)

    parameters = set(p for n, p in net.named_parameters()
                     if not n.endswith(".quantiles"))

    aux_parameters = set(p for n, p in net.named_parameters()
                         if n.endswith(".quantiles"))

    optimizer = optim.Adam(parameters, lr=opt.learning_rate)
    aux_optimizer = optim.Adam(aux_parameters, lr=opt.aux_learning_rate)
    begin_epoch = 0
    if opt.restore == True:
        checkpoint = torch.load(opt.ckpt)
        net.load_state_dict(checkpoint['state_dict'])
        aux_optimizer.load_state_dict(checkpoint['aux_optimizer'])
        begin_epoch = checkpoint['epoch']

    for epoch in range(begin_epoch, opt.epochs):
        if opt.save:
            net.update(force=True)
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict()
                },
                save_path=f"./ckpt/{opt.model}/{opt.label_str}/{str(config.args.lmbda)}/",
                filename="{:0>4d}.pth.tar".format(epoch)
            )

        train_one_epoch(
            opt,
            net,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch
        )

def test_img(model, x, p_x, save_name, lmbda, label):
    model.eval()
    x = x.unsqueeze(0)
    p_x = p_x.unsqueeze(0)

    x = x.cuda()
    p_x = p_x.cuda()

    h, w = x.size(2), x.size(3)
    p = 64  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0.0,
    )

    out_enc = model.compress(x_padded)

    save_path = (save_name.replace(label, label+'_bin')).replace('.png', '.bin')
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    write_to_bin(out_enc, h=h, w=w, save_path=save_path, lmbda=lmbda)
    bits_bin = os.path.getsize(save_path)
    bits_bin = bits_bin * 8
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    pixel_bits = bits_bin / num_pixels
    strings, original_size, shape, lmbda = read_from_bin(save_path)

    out_dec = model.decompress(strings, shape)

    out_dec["im_x_hat"] = F.pad(
        out_dec["im_x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )
    tensor2img(out_dec["im_x_hat"], save_name)

    return {
        "psnr": psnr(p_x, out_dec["im_x_hat"]),
        "bpp": pixel_bits
    }

def test(opt):
    device = "cuda" if opt.cuda and torch.cuda.is_available() else "cpu"

    lmbda_list = [0, 1, 2, 3, 4]
    ll = len(lmbda_list)
    result_array = np.zeros([2, ll])

    for jj in range(ll):  # range(0, 7): # range(0, 7):
        opt.lmbda = lmbda_list[jj]
        net = create_model(opt)
        net.to(device)
        
        if not len(opt.ckpt) == 0:
            ckpt_path = opt.ckpt
        else:
            ckpt_path = './ckpt/{model}/{idx}-best.pth.tar'.format(
                model=opt.model, idx = opt.lmbda)
        
        checkpoint = torch.load(ckpt_path)
        net.load_state_dict(checkpoint['state_dict'])
        net.eval()

        transform = transforms.Compose([transforms.ToTensor()])
        
        
        gt_img_list, de_img_list = data_load_same_name(opt.test_dataset_de, opt.test_dataset_gt)
        if not (len(gt_img_list) > 0 & len(de_img_list)):
            raise ValueError("Please check the dataset path! Or check the image whether it is loaded or not.")

        psnr_all = []
        bpp_all = []

        for img_path, p_img_path in zip(de_img_list, gt_img_list):

            img = Image.open(img_path).convert('RGB')
            image_or = Image.open(p_img_path).convert('RGB')
            
            basename = os.path.basename(img_path)
            save_name = f'./result/Ours/{opt.model}/{opt.label_str}/{lmbda_list[jj]}/{basename}'
            if not os.path.exists(os.path.dirname(save_name)):
                os.makedirs(os.path.dirname(save_name))

            result = test_img(
                net,
                transform(img),
                transform(image_or),
                save_name, lmbda_list[jj], opt.label_str)

            psnr_all.append(result['psnr'])
            bpp_all.append(result['bpp'])


            #result_str1 = '{}-----psnr:{:.2f}, bpp:{:.2f}'.format(
            #    opt.model,
            #    result['psnr'],
            #    result['bpp'])
            #print(result_str1)

        psnr_all = np.array(psnr_all)
        bpp_all = np.array(bpp_all)

        result_array[0, jj] = np.mean(psnr_all)
        result_array[1, jj] = np.mean(bpp_all)

        result_str = '{} psnr:{:.2f}, bpp:{:.2f}'.format(
            opt.model,
            np.mean(psnr_all),
            np.mean(bpp_all))
        print(result_str)

        psnr_all = np.array(psnr_all)
        bpp_all = np.array(bpp_all)

        result_array[0, jj] = np.mean(psnr_all)
        result_array[1, jj] = np.mean(bpp_all)

    print('----------result------')
    for ii in range(2):
        print(result_array[ii, :])
        print("\n")

    print('----------result------')
    for ii in range(2):
        result_str = ''
        for kk in range(ll):
            result_str += f'{result_array[ii, kk]}\t'
        print(result_str)
        print("\n")


if __name__ == "__main__":
    opt = config.args
    if opt.mode == 'train':
        train(opt)
    elif opt.mode == 'test':
        test(opt)