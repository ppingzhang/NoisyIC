import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# pylint: disable=E0611,E0401
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d
from compressai.models import CompressionModel
from compressai.models.utils import conv, deconv, update_registered_buffers

from loss.Vgg19 import VGG19_LossNetwork


# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

def conv1(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

class ca_layer(nn.Module):
    def __init__(self, channel, reduction=8, bias=True):
        super(ca_layer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class DAU(nn.Module):
    def __init__(
            self, n_feat, kernel_size=3, reduction=8,
            bias=False, bn=False, act=nn.PReLU(), res_scale=1):
        super(DAU, self).__init__()
        modules_body = [conv1(n_feat, n_feat, kernel_size, bias=bias), act, conv1(n_feat, n_feat, kernel_size, bias=bias)]
        self.body = nn.Sequential(*modules_body)

        ## Spatial Attention
        self.SA = spatial_attn_layer()

        ## Channel Attention
        self.CA = ca_layer(n_feat, reduction, bias=bias)

        self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        res = self.body(x)
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        res += x
        return res

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ResBlock, self).__init__()
        self.in_ch = int(in_channel)
        self.out_ch = int(out_channel)
        self.k = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

        self.conv1 = nn.Conv2d(self.in_ch, self.out_ch, self.k, self.stride, self.padding)
        self.conv2 = nn.Conv2d(self.in_ch, self.out_ch, self.k, self.stride, self.padding)

        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv2(self.relu(self.conv1(x)))
        out = x + x1
        return out

# Layer one: enhancement
class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=5):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale

class atten_ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(atten_ResBlock, self).__init__()
        self.in_ch = int(in_channel)
        self.out_ch = int(out_channel)
        self.k = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

        self.conv1 = nn.Conv2d(self.in_ch, self.out_ch, self.k, self.stride, self.padding)
        self.satten = spatial_attn_layer()
        self.conv2 = nn.Conv2d(self.in_ch, self.out_ch, self.k, self.stride, self.padding)

        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv2(self.satten(self.relu(self.conv1(x))))
        out = x + x1
        return out

# Layer two: adaptive denoising
class SKFF(nn.Module):
    def __init__(self, in_channels, height=3, reduction=8, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V


'''
1. H-net 中的conv 用WV-trans替代
'''


class MainCodec(CompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, opt=None, lmbda=1, **kwargs):
        

        ####
        if opt.label_str == '64_96':
            self.N = 128//2 
            self.M = 192//2
        elif opt.label_str == '128_96':
            self.N = 128
            self.M = 192//2
        else:
            self.N = 128 
            self.M = 192

        super().__init__(entropy_bottleneck_channels=self.N, **kwargs)


        self.en_pos0_cov = nn.Conv2d(3, self.N, 3, 1, 1)

        self.en_pos1_res = atten_ResBlock(self.N, self.N, 3, 1, 1)
        self.en_pos1_avp = torch.nn.AvgPool2d(5, stride=1, padding=2)
        self.en_pos1_skf = SKFF(self.N, 2)
        self.en_pos1_dwn = nn.Conv2d(self.N, self.N * 2, 3, 2, 1)

        self.en_pos2_res = atten_ResBlock(self.N, self.N, 3, 1, 1)
        self.en_pos2_avp = torch.nn.AvgPool2d(5, stride=1, padding=2)
        self.en_pos2_skf = SKFF(self.N, 2)
        self.en_pos2_dwn = nn.Conv2d(self.N, self.N * 2, 3, 2, 1)

        self.en_pos3_res = atten_ResBlock(self.N, self.N, 3, 1, 1)
        self.en_pos3_avp = torch.nn.AvgPool2d(5, stride=1, padding=2)
        self.en_pos3_skf = SKFF(self.N, 2)
        self.en_pos3_dwn = nn.Conv2d(self.N, self.N * 2, 3, 2, 1)

        self.en_pos4_res = atten_ResBlock(self.N, self.N, 3, 1, 1)
        self.en_pos4_avp = torch.nn.AvgPool2d(5, stride=1, padding=2)
        self.en_pos4_skf = SKFF(self.N, 2)
        self.en_pos4_dwn = nn.Conv2d(self.N, self.N * 2, 3, 2, 1)

        self.en_pos_all = nn.Conv2d(self.N, self.M, 3, 1, 1)

        self.de_pos0 = nn.Conv2d(self.M, self.N, 1, 1, 0)
        self.de_pos1 = nn.Sequential(ResBlock(self.N, self.N, 3, 1, 1), ResBlock(self.N, self.N, 3, 1, 1),
                                     ResBlock(self.N, self.N, 3, 1, 1), nn.ConvTranspose2d(self.N, self.N, 5, 2, 2, 1))
        self.de_pos2 = nn.Sequential(ResBlock(self.N, self.N, 3, 1, 1), ResBlock(self.N, self.N, 3, 1, 1),
                                     ResBlock(self.N, self.N, 3, 1, 1), nn.ConvTranspose2d(self.N, self.N, 5, 2, 2, 1))
        self.de_pos3 = nn.Sequential(ResBlock(self.N, self.N, 3, 1, 1), ResBlock(self.N, self.N, 3, 1, 1),
                                     ResBlock(self.N, self.N, 3, 1, 1), nn.ConvTranspose2d(self.N, self.N, 5, 2, 2, 1))
        self.de_pos4 = nn.Sequential(ResBlock(self.N, self.N, 3, 1, 1), ResBlock(self.N, self.N, 3, 1, 1),
                                     ResBlock(self.N, self.N, 3, 1, 1), nn.ConvTranspose2d(self.N, 3, 5, 2, 2, 1))


        self.de1_neg1 = nn.Sequential(ResBlock(self.N, self.N, 3, 1, 1), ResBlock(self.N, self.N, 3, 1, 1),
                                     ResBlock(self.N, self.N, 3, 1, 1), nn.ConvTranspose2d(self.N, self.N, 5, 2, 2, 1))
        self.de1_neg2 = nn.Sequential(ResBlock(self.N, self.N, 3, 1, 1), ResBlock(self.N, self.N, 3, 1, 1),
                                     ResBlock(self.N, self.N, 3, 1, 1), nn.ConvTranspose2d(self.N, self.N, 5, 2, 2, 1))
        self.de1_neg3 = nn.Sequential(ResBlock(self.N, self.N, 3, 1, 1), ResBlock(self.N, self.N, 3, 1, 1),
                                     ResBlock(self.N, self.N, 3, 1, 1), nn.ConvTranspose2d(self.N, self.N, 5, 2, 2, 1))
        self.de1_neg4 = nn.Sequential(ResBlock(self.N, self.N, 3, 1, 1), ResBlock(self.N, self.N, 3, 1, 1),
                                     ResBlock(self.N, self.N, 3, 1, 1), nn.ConvTranspose2d(self.N, 3, 5, 2, 2, 1))


        self.de2_neg1 = nn.Sequential(ResBlock(self.N, self.N, 3, 1, 1), ResBlock(self.N, self.N, 3, 1, 1),
                                     ResBlock(self.N, self.N, 3, 1, 1), nn.ConvTranspose2d(self.N, self.N, 5, 2, 2, 1))
        self.de2_neg2 = nn.Sequential(ResBlock(self.N, self.N, 3, 1, 1), ResBlock(self.N, self.N, 3, 1, 1),
                                     ResBlock(self.N, self.N, 3, 1, 1), nn.ConvTranspose2d(self.N, self.N, 5, 2, 2, 1))
        self.de2_neg3 = nn.Sequential(ResBlock(self.N, self.N, 3, 1, 1), ResBlock(self.N, self.N, 3, 1, 1),
                                     ResBlock(self.N, self.N, 3, 1, 1), nn.ConvTranspose2d(self.N, self.N, 5, 2, 2, 1))
        self.de2_neg4 = nn.Sequential(ResBlock(self.N, self.N, 3, 1, 1), ResBlock(self.N, self.N, 3, 1, 1),
                                     ResBlock(self.N, self.N, 3, 1, 1), nn.ConvTranspose2d(self.N, 3, 5, 2, 2, 1))


        self.h_a1 = nn.Sequential(
            conv(self.M, self.N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(self.N, self.N),
            nn.LeakyReLU(inplace=True),
            conv(self.N, self.M),
        )

        self.h_s1 = nn.Sequential(
            deconv(self.M, self.M),
            nn.LeakyReLU(inplace=True),
            deconv(self.M, self.M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(self.M * 3 // 2, self.M * 2, stride=1, kernel_size=3),
        )

        self.h_a2 = nn.Sequential(
            conv(self.M, self.N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(self.N, self.N),
            nn.ReLU(inplace=True),
            conv(self.N, self.N),
        )

        self.h_s2 = nn.Sequential(
            deconv(self.N, self.N),
            nn.ReLU(inplace=True),
            deconv(self.N, self.N),
            nn.ReLU(inplace=True),
            conv(self.N, self.M*2, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.gaussian_conditional1 = GaussianConditional(None)

        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss(reduction='mean')
        self.vgg = VGG19_LossNetwork()

        self.kk = 0

    def triplet_loss(self, anchor, positive, negative, margin=1.0):
        tl = torch.mean(
            torch.max(torch.abs(anchor - positive) - torch.abs(anchor - negative) + margin, torch.zeros_like(anchor)))
        return tl

    def forward(self, x1, x2, is_training=True):
        # print(x1.shape)
        f1_all0 = self.en_pos0_cov(x1)
        f1_res1 = self.en_pos1_res(f1_all0)  
        f1_avg1 =  self.en_pos1_avp(f1_all0)
        f1_all1 = self.en_pos1_dwn(self.en_pos1_skf([f1_res1, f1_avg1]))
        f1_p1 = f1_all1[:, 0:self.N, :, :]

        f1_res2 = self.en_pos2_res(f1_p1) 
        f1_avg2 = self.en_pos2_avp(f1_p1)
        f1_all2 = self.en_pos2_dwn(self.en_pos2_skf([f1_res2, f1_avg2]))
        f1_p2 = f1_all2[:, 0:self.N, :, :]

        f1_res3 = self.en_pos3_res(f1_p2) 
        f1_avg3 = self.en_pos3_avp(f1_p2)
        f1_all3 = self.en_pos3_dwn(self.en_pos3_skf([f1_res3, f1_avg3]))
        f1_p3 = f1_all3[:, 0:self.N, :, :]

        f1_res4 = self.en_pos4_res(f1_p3) 
        f1_avg4 = self.en_pos4_avp(f1_p3)
        f1_all4 = self.en_pos4_dwn(self.en_pos4_skf([f1_res4, f1_avg4]))
        f1_p4 = f1_all4[:, 0:self.N, :, :]

        y = self.en_pos_all(f1_p4)

        z = self.h_a1(y)
        zz = self.h_a2(torch.abs(z))
        zz_hat, zz_likelihoods = self.entropy_bottleneck(zz)
        gaussian_params_zz = self.h_s2(zz_hat)
        scales_hat_zz, means_hat_zz = gaussian_params_zz.chunk(2, 1)
        z_hat, z_likelihoods = self.gaussian_conditional(z, scales_hat_zz, means=means_hat_zz)
        gaussian_params_z = self.h_s1(z_hat)
        scales_hat_z, means_hat_z = gaussian_params_z.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional1(y, scales_hat_z, means=means_hat_z)

        
        y1_hat0 = self.de_pos0(y_hat)
        y1_hat1 = self.de_pos1(y1_hat0)
        y1_hat2 = self.de_pos2(y1_hat1)
        y1_hat3 = self.de_pos3(y1_hat2)
        x_hat = self.de_pos4(y1_hat3)

        if is_training:

            f2_all0 = self.en_pos0_cov(x2)
            f2_res1 = self.en_pos1_res(f2_all0)  
            f2_avg1 = self.en_pos1_avp(f2_all0)
            f2_all1 = self.en_pos1_dwn(self.en_pos1_skf([f2_res1, f2_avg1]))
            f2_p1 = f2_all1[:, 0:self.N, :, :]

            f2_res2 = self.en_pos2_res(f2_p1)  
            f2_avg2 = self.en_pos2_avp(f2_p1)
            f2_all2 = self.en_pos2_dwn(self.en_pos2_skf([f2_res2, f2_avg2]))
            f2_p2 = f2_all2[:, 0:self.N, :, :]

            f2_res3 = self.en_pos3_res(f2_p2)  
            f2_avg3 = self.en_pos3_avp(f2_p2)
            f2_all3 = self.en_pos3_dwn(self.en_pos3_skf([f2_res3, f2_avg3]))
            f2_p3 = f2_all3[:, 0:self.N, :, :]

            f2_res4 = self.en_pos4_res(f2_p3)  
            f2_avg4 = self.en_pos4_avp(f2_p3)
            f2_all4 = self.en_pos4_dwn(self.en_pos4_skf([f2_res4, f2_avg4]))
            f2_p4 = f2_all4[:, 0:self.N, :, :]

            y2 = self.en_pos_all(f2_p4)

            z2 = self.h_a1(y2)
            zz2 = self.h_a2(torch.abs(z2))
            zz2_hat, zz2_likelihoods = self.entropy_bottleneck(zz2)
            gaussian_params_zz2 = self.h_s2(zz2_hat)
            scales_hat_zz2, means_hat_zz2 = gaussian_params_zz2.chunk(2, 1)
            z2_hat, z2_likelihoods = self.gaussian_conditional(z2, scales_hat_zz2, means=means_hat_zz2)
            gaussian_params_z2 = self.h_s1(z2_hat)
            scales_hat_z2, means_hat_z2 = gaussian_params_z2.chunk(2, 1)
            y2_hat, y2_likelihoods = self.gaussian_conditional1(y2, scales_hat_z2, means=means_hat_z2)


            y2_hat0 = self.de_pos0(y2_hat)
            y2_hat1 = self.de_pos1(y2_hat0)
            y2_hat2 = self.de_pos2(y2_hat1)
            y2_hat3 = self.de_pos3(y2_hat2)
            x2_hat = self.de_pos4(y2_hat3)

            f1_n1 = f1_p1 + f1_all1[:, self.N:self.N * 2, :, :]
            f1_n2 = f1_p2 + f1_all2[:, self.N:self.N * 2, :, :]
            f1_n3 = f1_p3 + f1_all3[:, self.N:self.N * 2, :, :]
            f1_n4 = f1_p4 + f1_all4[:, self.N:self.N * 2, :, :]

            de_f1_m1 = self.de1_neg1(f1_n4)
            de_f1_m2 = self.de1_neg2(de_f1_m1)
            de_f1_m3 = self.de1_neg3(de_f1_m2)
            x_n1_hat = self.de1_neg4(de_f1_m3)

            f2_n1 = f2_p1 + f2_all1[:, self.N:self.N * 2, :, :]
            f2_n2 = f2_p2 + f2_all2[:, self.N:self.N * 2, :, :]
            f2_n3 = f2_p3 + f2_all3[:, self.N:self.N * 2, :, :]
            f2_n4 = f2_p4 + f2_all4[:, self.N:self.N * 2, :, :]

            de_f2_m1 = self.de2_neg1(f2_n4)
            de_f2_m2 = self.de2_neg2(de_f2_m1)
            de_f2_m3 = self.de2_neg3(de_f2_m2)
            x_n2_hat = self.de2_neg4(de_f2_m3)

            l1_1 = self.triplet_loss(f1_p1, f2_p1, f2_n1, margin=1.0)
            l1_2 = self.triplet_loss(f1_p2, f2_p2, f2_n2, margin=1.0)
            l1_3 = self.triplet_loss(f1_p3, f2_p3, f2_n3, margin=1.0)
            l1_4 = self.triplet_loss(f1_p4, f2_p4, f2_n4, margin=1.0)

            l2_1 = self.triplet_loss(f2_p1, f1_p1, f1_n1, margin=1.0)
            l2_2 = self.triplet_loss(f2_p2, f1_p2, f1_n2, margin=1.0)
            l2_3 = self.triplet_loss(f2_p3, f1_p3, f1_n3, margin=1.0)
            l2_4 = self.triplet_loss(f2_p4, f1_p4, f1_n4, margin=1.0)

            loss_tl = (l1_1 + l1_2 + l1_3 + l1_4 + l2_1 + l2_2 + l2_3 + l2_4) / 8.0

            return {
                "im_x1_hat": x_hat,
                "im_x2_hat": x2_hat,
                "im_n1_hat": x_n1_hat,
                "im_n2_hat": x_n2_hat,
                "likelihoods1": {"y": y_likelihoods, "z": z_likelihoods, 'zz':zz_likelihoods},
                "likelihoods2": {"y": y2_likelihoods, "z": z2_likelihoods, 'zz':zz2_likelihoods},
                "loss_tl": loss_tl
            }
        else:
            fn_1 = f1_all1[:, self.N:self.N * 2, :, :]
            fn_2 = f1_all2[:, self.N:self.N * 2, :, :]
            fn_3 = f1_all3[:, self.N:self.N * 2, :, :]
            fn_4 = f1_all4[:, self.N:self.N * 2, :, :]

            return {
                "fn_1": fn_1, 
                "fn_2": fn_2, 
                "fn_3": fn_3, 
                "fn_4": fn_4, 
                "fp_1": f1_p1, 
                "fp_2": f1_p2, 
                "fp_3": f1_p3, 
                "fp_4": f1_p4, 
                "y":y,
                "scales_hat_zz":scales_hat_zz,
                "means_hat_zz":means_hat_zz,
                "zz_hat":zz_hat,
                "scales_hat_z":scales_hat_z,
                "means_hat_z":means_hat_z,
                "z_hat":z_hat,
                "y_hat":y_hat,
                "zz":zz,
                "z": z,
                "im_x_hat": x_hat,
                "likelihoods": {"y": y_likelihoods, "z": z_likelihoods, "zz": zz_likelihoods}
            }

    def load_state_dict(self, state_dict):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional1,
            "gaussian_conditional1",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )

        super().load_state_dict(state_dict)

    def compress(self, x):

        f1_all0 = self.en_pos0_cov(x)
        f1_res1 = self.en_pos1_res(f1_all0)  # postive layer1
        f1_avg1 =  self.en_pos1_avp(f1_all0)
        f1_all1 = self.en_pos1_dwn(self.en_pos1_skf([f1_res1, f1_avg1]))
        f1_p1 = f1_all1[:, 0:self.N, :, :]

        f1_res2 = self.en_pos2_res(f1_p1)  # postive layer1
        f1_avg2 = self.en_pos2_avp(f1_p1)
        f1_all2 = self.en_pos2_dwn(self.en_pos2_skf([f1_res2, f1_avg2]))
        f1_p2 = f1_all2[:, 0:self.N, :, :]

        f1_res3 = self.en_pos3_res(f1_p2)  # postive layer1
        f1_avg3 = self.en_pos3_avp(f1_p2)
        f1_all3 = self.en_pos3_dwn(self.en_pos3_skf([f1_res3, f1_avg3]))
        f1_p3 = f1_all3[:, 0:self.N, :, :]

        f1_res4 = self.en_pos4_res(f1_p3)  # postive layer1
        f1_avg4 = self.en_pos4_avp(f1_p3)
        f1_all4 = self.en_pos4_dwn(self.en_pos4_skf([f1_res4, f1_avg4]))
        f1_p4 = f1_all4[:, 0:self.N, :, :]

        y = self.en_pos_all(f1_p4)

        z = self.h_a1(y)
        zz = self.h_a2(torch.abs(z))

        zz_strings = self.entropy_bottleneck.compress(zz)
        zz_hat = self.entropy_bottleneck.decompress(zz_strings, zz.size()[-2:])
        gaussian_params_zz = self.h_s2(zz_hat)
        scales_hat_zz, means_hat_zz = gaussian_params_zz.chunk(2, 1)

        indexes_z = self.gaussian_conditional.build_indexes(scales_hat_zz)
        z_strings = self.gaussian_conditional.compress(z, indexes_z, means=means_hat_zz)
        z_hat = self.gaussian_conditional.decompress(z_strings, indexes_z, means=means_hat_zz)

        gaussian_params = self.h_s1(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional1.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional1.compress(y, indexes, means=means_hat)

        return {"strings": [y_strings, z_strings, zz_strings], "shape": zz.size()[-2:]}

    def decompress(self, strings, shape):

        zz_hat = self.entropy_bottleneck.decompress(strings[2], shape)
        gaussian_params_zz = self.h_s2(zz_hat)
        scales_hat_zz, means_hat_zz = gaussian_params_zz.chunk(2, 1)

        indexes_zz = self.gaussian_conditional.build_indexes(scales_hat_zz)
        z_hat = self.gaussian_conditional.decompress(strings[1], indexes_zz, means=means_hat_zz)

        gaussian_params = self.h_s1(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional1.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional1.decompress(strings[0], indexes, means=means_hat)

        y_hat0 = self.de_pos0(y_hat)
        y_hat1 = self.de_pos1(y_hat0)
        y_hat2 = self.de_pos2(y_hat1)
        y_hat3 = self.de_pos3(y_hat2)
        x_hat  = self.de_pos4(y_hat3).clamp_(0, 1)  # decode image 1

        return {"im_x_hat": x_hat}

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated1 = self.gaussian_conditional1.update_scale_table(scale_table, force=force)
        updated |= updated1
        updated |= super().update(force=force)
        return updated

    def loss(self, output, p_im=None, noise_img1=None, noise_img2=None, lmbda=1):

        N, _, H, W = p_im.size()
        num_pixels = N * H * W
        out = {}
        out["bpp_loss1"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods1"].values())

        out["bpp_loss2"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods2"].values())

        out["mse_im1"] = float(lmbda) * 256 * self.mse(output["im_x1_hat"], p_im)
        out["mse_im2"] = float(lmbda) * 256 * self.mse(output["im_x2_hat"], p_im)

        # print(noise_img1 - p_im)
        out["mse_n1"] = float(lmbda) * 256 * self.mse(output["im_n1_hat"], noise_img1) * 0.1
        out["mse_n2"] = float(lmbda) * 256 * self.mse(output["im_n2_hat"], noise_img2) * 0.1
        out["loss_tl"] = float(lmbda) * 256 * output["loss_tl"]

        vgg_list1 = self.vgg(output["im_x1_hat"], p_im)  # vgg 1.0 vgg1 0.5
        vgg_list2 = self.vgg(output["im_x2_hat"], p_im)  # vgg 1.0 vgg1 0.5
        out["vgg1"] = float(lmbda) * 256 * vgg_list1[0] * 0.5
        out["vgg2"] = float(lmbda) * 256 * vgg_list2[0] * 0.5

        out["loss"] = out["mse_im1"] + out["mse_im2"] + out["mse_n1"] + out["mse_n2"] + out["loss_tl"] + out["vgg1"] + \
                      out["vgg2"] + out["bpp_loss1"] + out["bpp_loss2"]

        return out

    def test_loss(self, output, p_im, lmbda):

        N, _, H, W = p_im.size()
        num_pixels = N * H * W
        out = {}
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        out["mse_loss"] = 10 * math.log10(255.0 ** 2 / self.mse(output["im_x_hat"], p_im))
        out["loss"] = out["mse_loss"] / out["bpp_loss"] 
        return out



