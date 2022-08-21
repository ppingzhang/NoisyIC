
from PIL import Image
import os
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data.dataset import Dataset
import random
import torch
import pdb
import glob
import torchvision.transforms.functional as TF

from data.data_util import AddGaussianNoise
from data.dataset_load import load_train_dataset, SIDD_patch_load
import random
import numpy as np



class TrainDataset_Mix_Noise(Dataset):
	def __init__(self, data_list, image_size=256):

		self.image_size = image_size
		self.variance_list = [1, 10, 20, 30, 40, 50]
		self.image_path = load_train_dataset(data_list)
		self.gt_img_list, self.de_img_list = SIDD_patch_load()


	# random.shuffle(self.noise_map_path)

	def transform_same(self, gt, im1, im2, crop_size=(256, 256)):

		# Random crop
		i, j, h, w = transforms.RandomCrop.get_params(
			gt, output_size=crop_size)
		ss = gt.size
		rh = random.randint(0, ss[0]-crop_size[0])
		rw = random.randint(0, ss[1]-crop_size[1])

		gt = gt.crop((rh, rw, rh+crop_size[0], rw+crop_size[1]))
		im1 = im1.crop((rh, rw, rh+crop_size[0], rw+crop_size[1]))
		im2 = im2.crop((rh, rw, rh+crop_size[0], rw+crop_size[1]))

		if random.random() > 0.5:
			gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
			im1 = im1.transpose(Image.FLIP_LEFT_RIGHT)
			im2 = im2.transpose(Image.FLIP_LEFT_RIGHT)

		# Random vertical flipping
		if random.random() > 0.5:
			gt = gt.transpose(Image.FLIP_TOP_BOTTOM)
			im1 = im1.transpose(Image.FLIP_TOP_BOTTOM)
			im2 = im2.transpose(Image.FLIP_TOP_BOTTOM)


		return gt, im1, im2

	def __getitem__(self, item):
		seed = np.random.randint(2147483647)
		random.seed(seed)

		transform2 = transforms.Compose([
			transforms.ToTensor()
		])

		if random.random() > 0.6:


			idx = item % len(self.gt_img_list)
			im_gt_path = self.gt_img_list[idx]
			im_gt = Image.open(im_gt_path).convert('RGB')

			im_de_path = self.de_img_list[idx]
			im_noise1 = Image.open(im_de_path).convert('RGB')

			im_noise2 = AddGaussianNoise(im_gt, variance=random.choice(self.variance_list))
			patch_gt, patch_n1, patch_n2 = self.transform_same(im_gt, im_noise1, im_noise2)


		else:
			im_gt_path = self.image_path[item]
			im_gt = Image.open(im_gt_path).convert('RGB')

			im_noise1 = AddGaussianNoise(im_gt, variance=random.choice(self.variance_list))
			im_noise2 = AddGaussianNoise(im_gt, variance=random.choice(self.variance_list))

			patch_gt, patch_n1, patch_n2 = self.transform_same(im_gt, im_noise1, im_noise2)

		patch_train_gt = transform2(patch_gt)
		patch_train_n1 = transform2(patch_n1)
		patch_train_n2 = transform2(patch_n2)

		return patch_train_gt, patch_train_n1, patch_train_n2

	def __len__(self):
		return len(self.image_path)

def DataLoader_Noise_domain_noise(train_list, image_size, batch_size):
	train_dataset = TrainDataset_Mix_Noise(train_list, image_size)

	train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
											   batch_size=batch_size,
											   shuffle=True, num_workers=5)

	return train_loader







if __name__ == '__main__':
	pass