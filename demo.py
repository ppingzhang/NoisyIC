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
import argparse


import config as config

from Common.write_bin import read_from_bin, write_to_bin
from data.dataset_noise_mix import DataLoader_Noise_domain_noise


from model import create_model
from data.test_dataload import data_load, data_load_same_name

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.backends.cudnn.deterministic = True

def test_img(model, x,  img_save_path, bin_save_path, lmbda):
    model.eval()
    x = x.unsqueeze(0)
    x = x.cuda()

    h, w = x.size(2), x.size(3)
    p = 128  # maximum 6 strides of 2
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

    
    if not os.path.exists(os.path.dirname(img_save_path)):
        os.makedirs(os.path.dirname(img_save_path))
    if not os.path.exists(os.path.dirname(bin_save_path)):
        os.makedirs(os.path.dirname(bin_save_path))

    write_to_bin(out_enc, h=h, w=w, save_path=bin_save_path, lmbda=lmbda)
    bits_bin = os.path.getsize(bin_save_path)
    bits_bin = bits_bin * 8
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    pixel_bits = bits_bin / num_pixels
    strings, original_size, shape, lmbda = read_from_bin(bin_save_path)

    out_dec = model.decompress(strings, shape)

    out_dec["im_x_hat"] = F.pad(
        out_dec["im_x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )
    tensor2img(out_dec["im_x_hat"], img_save_path)

    return pixel_bits

def test(opt, img_path, ckpt_path, lmbda, img_save_path, bin_save_path):
    device = "cuda" if opt.cuda and torch.cuda.is_available() else "cpu"

    net = create_model(opt)
    net.to(device)
    
    checkpoint = torch.load(ckpt_path)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()

    transform = transforms.Compose([transforms.ToTensor()])

    img = Image.open(img_path).convert('RGB')
    
    bpp = test_img(net, transform(img), img_save_path, bin_save_path, lmbda)
    print(f"bpp: {bpp}")



if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="End2End Image Compression Settings")

    parser.add_argument('--model', default="MainCodec", type=str, help='')
    parser.add_argument('--label_str', default="basic", type=str, help='')
    parser.add_argument('--cuda', action='store_false', default=True, help='Use cuda')

    parser.add_argument('--lmbda', dest='lmbda', type=int, default=1, help='')
    parser.add_argument('--ckpt', default="ckpt/MainCodec/1/best.pth.tar", type=str, help='checkpoint')
    parser.add_argument('--img_path', default="dataset/train/flickr3k/999233601_6d9c206eea_b.jpg", type=str, help='noisy image path')
    parser.add_argument('--img_save_path', default="./tmp.jpg", type=str, help='decided image path')
    parser.add_argument('--bin_save_path', default="./tmp.bin", type=str, help='bin path') 
    args = parser.parse_args()
    

    test(args, args.img_path, args.ckpt, args.lmbda, args.img_save_path, args.bin_save_path)