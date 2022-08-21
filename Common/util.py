

import os
import math
import shutil

import torch
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter
from torchvision import transforms

def save_checkpoint(state, save_path, filename="checkpoint.pth.tar", is_best=False):
	
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	torch.save(state, save_path+filename)
	if is_best:
		shutil.copyfile(save_path+filename, save_path+"/best.pth.tar")

def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
	mse = F.mse_loss(a, b).item()
	return -10 * math.log10(mse)

def tensor2img(tensor_im, save_im):
	unloader = transforms.ToPILImage()
	image = tensor_im.cpu().clone()  
	image = image.squeeze(0) 
	image = torch.clamp(image, 0, 1)
	image = unloader(image)

	dpath = os.path.dirname(save_im) 
	
	if not os.path.exists(dpath):
		os.makedirs(dpath)
		
	image.save(save_im)


def print_loss(output):
	str_p = ""
	for k in output:
		str_p += f'\t{k}: {output[k].item():.5f} |'
	return str_p

			

def show_in_board(writer, output, step, **imgs):

	for k in imgs:
		in_clamp = torch.clamp(imgs[k], 0, 1)
		in_grid = torchvision.utils.make_grid(in_clamp)
		writer.add_image(k, in_grid, step)

	for k in output:
		if 'im_' in k:
			im_clamp = torch.clamp(output[k], 0, 1)
			im_grid = torchvision.utils.make_grid(im_clamp)
			writer.add_image(k, im_grid, step)
	
