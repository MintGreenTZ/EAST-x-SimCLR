import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from torchvision import transforms
from dataset import custom_dataset
from PIL import Image
from model import EAST
from loss import Loss
from gaussian_blur import GaussianBlur
import os
import time
import numpy as np


def train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, interval, s):
	img_files = os.listdir(train_img_path)
	img_files = sorted([os.path.join(train_img_path, img_file) for img_file in img_files])

	pic_size = 512

	color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
	data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=pic_size),
										  transforms.RandomHorizontalFlip(),
										  transforms.RandomApply([color_jitter], p=0.8),
										  transforms.RandomGrayscale(p=0.2),
										  GaussianBlur(kernel_size=int(0.1 * pic_size)),
										  transforms.ToTensor()])
	img2tensor = transforms.Compose([transforms.ToTensor()])
	print("transforms done")

	file_num = len(os.listdir(train_img_path))
	trainset = custom_dataset(train_img_path, train_gt_path)
	train_loader = data.DataLoader(trainset, batch_size=batch_size, \
                                   shuffle=True, num_workers=num_workers, drop_last=True)
	
	criterion = Loss()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = EAST()
	data_parallel = False
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		data_parallel = True
	model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[epoch_iter//2], gamma=0.1)

	# for i, img_file in enumerate(img_files):
	# 	im = Image.open(img_file)
	# 	print(img2tensor(im))
	# 	data_transforms(im)
	# 	print(img2tensor(im))
	# 	print(data_transforms(im))
	# 	data_transforms(im)
	# 	print(img2tensor(im))
	# 	print(data_transforms(im))
	# 	print(data_transforms(im).size())
	# 	if i == 2:
	# 		break
	#
	# print("img finished")


	for epoch in range(epoch_iter):	
		model.train()
		scheduler.step()
		epoch_loss = 0
		epoch_time = time.time()
		for i, (img1, gt_score1, gt_geo1, ignored_map1,
				img2, gt_score2, gt_geo2, ignored_map2) in enumerate(train_loader):
			# print(img.size())
			# print(gt_score.size())
			# print(gt_geo.size())
			# print(ignored_map.size())

			start_time = time.time()


			img1, gt_score1, gt_geo1, ignored_map1 = img1.to(device), gt_score1.to(device), gt_geo1.to(device), ignored_map1.to(device)
			pred_score, pred_geo = model(img1)
			loss = criterion(gt_score1, pred_score, gt_geo1, pred_geo, ignored_map1)
			
			epoch_loss += loss.item()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()


			img2, gt_score2, gt_geo2, ignored_map2 = img2.to(device), gt_score2.to(device), gt_geo2.to(
				device), ignored_map2.to(device)
			pred_score, pred_geo = model(img2)
			loss = criterion(gt_score2, pred_score, gt_geo2, pred_geo, ignored_map2)

			epoch_loss += loss.item()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(\
              epoch+1, epoch_iter, i+1, int(file_num/batch_size), time.time()-start_time, loss.item()))
		
		print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss/int(file_num/batch_size), time.time()-epoch_time))
		print(time.asctime(time.localtime(time.time())))
		print('='*50)
		if (epoch + 1) % interval == 0:
			state_dict = model.module.state_dict() if data_parallel else model.state_dict()
			torch.save(state_dict, os.path.join(pths_path, 'model_epoch_{}.pth'.format(epoch+1)))


if __name__ == '__main__':
	train_img_path = os.path.abspath('../ICDAR_2015/train_img')
	train_gt_path  = os.path.abspath('../ICDAR_2015/train_gt')
	pths_path      = './pths'
	batch_size     = 24 
	lr             = 1e-3
	num_workers    = 4
	epoch_iter     = 600
	save_interval  = 5
	s 			   = 1
	train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, save_interval, s)
	
