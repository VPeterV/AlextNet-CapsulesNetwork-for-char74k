# -*- coding: utf-8 -*-
# @Time    : 2020-06-25 14:41
# @Author  : PeterV
# @FileName: dataProcessor.py
# @Software: PyCharm

import os
import torch
from PIL import Image
from torch.utils.data import dataloader,dataset
from torchvision import datasets,transforms
from sklearn.model_selection import train_test_split


class DataProcessor:
	def __init__(self,batch_size,dataset='MNIST',data_path=None):
		self.data_path = data_path
		if dataset not in ['MNIST','char74k']:
			raise AttributeError("Sorry, unsupported dataset (only support for MNIST, char74k)")
		if dataset == 'MNIST':
			size = 28
		else:
			size = 20
		self.dataset_transform = transforms.Compose([transforms.Scale(size),
													 transforms.CenterCrop((size,size)),
												transforms.ToTensor(),
												transforms.Normalize((0.1307,),(0.3081,))	# data augmentation
												])
		self.target_transform = transforms.Compose([transforms.ToTensor()])
		if dataset == 'MNIST':
			train_dataset,test_dataset = self.MNIST_helper()
			valid_dataset = test_dataset
		else:
			train_dataset,valid_dataset,test_dataset = self.char74k_helper()

		self.train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
		self.valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=batch_size,shuffle=False)
		self.test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

	def char74k_helper(self):
		train_dataset = datasets.ImageFolder(root=self.data_path+'/train',transform=self.dataset_transform)
		valid_dataset = datasets.ImageFolder(root=self.data_path+'/dev',transform=self.dataset_transform)
		test_dataset = datasets.ImageFolder(root=self.data_path+'/test',transform=self.dataset_transform)

		# print(train_dataset)
		return train_dataset,valid_dataset,test_dataset


	def MNIST_helper(self):
		train_dataset = datasets.MNIST('./data', train=True, download=True, transform=self.dataset_transform)
		test_dataset = datasets.MNIST('./data', train=False, download=True, transform=self.dataset_transform)

		return train_dataset,test_dataset

if __name__ == '__main__':
	dataProcessor = DataProcessor(batch_size=128,dataset='char74k',data_path='data/char74k_preprocessed')
	for batch_id, (data,target) in enumerate(dataProcessor.train_loader):
		print('batch id:{} || data: {} || target: {}'.format(batch_id,data,target))
