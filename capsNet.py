# -*- coding: utf-8 -*-
# @Time    : 2020-06-25 14:41
# @Author  : PeterV
# @FileName: capsNet.py
# @Software: PyCharm

import torch
import torch.nn as nn
from modules.convLayer import ConvLayer
from modules.alexNet import AlexNet
from modules.primaryCaps import PrimaryCaps
from modules.digitCaps import DigitCaps
from modules.decoder import Decoder

class CapsNet(nn.Module):
	def __init__(self,conv_in=1,conv_out=256,conv_kernel=9,pri_cap_num=8,pri_out=256,pri_kernel=3,
				 pri_stride=2,digit_in=8,digit_out=16,num_classes=26,num_routes=256*2*2,share_weights=False,
				 decoder_in=16*10,de_outputI=512,de_outputII=1024,de_outputIII=1200,device=torch.device('cuda')):
		super(CapsNet, self).__init__()

		decoder_in = decoder_in // 10 * num_classes
		# print(decoder_in)
		self.device = device
		# self.convLayer = ConvLayer(conv_in,conv_out,conv_kernel)
		self.alexNetLayer = AlexNet(num_classes)
		self.primary_capsLayer = PrimaryCaps(pri_cap_num,conv_out,pri_out,pri_kernel,pri_stride)
		self.digit_capsLayer = DigitCaps(digit_in,digit_out,num_classes,num_routes,share_weights)
		self.decoder = Decoder(decoder_in,de_outputI,de_outputII,de_outputIII,num_classes)
		self.dropout = nn.Dropout(p=0.3)
		self.batch_norm = nn.BatchNorm2d(num_features=3)
		self.relu = nn.ReLU() 	# for computing loss
		self.mse_loss = nn.MSELoss()

	def  forward(self,input:torch.Tensor):
		# conv_output = self.convLayer(input)
		conv_output = self.alexNetLayer(input)
		# print(conv_output.shape)
		# conv_output = self.dropout(conv_output)
		# primaryCaps_output = self.primary_capsLayer(conv_output)
		# primaryCaps_output = self.dropout(primaryCaps_output)
		# digitCaps_output = self.digit_capsLayer(primaryCaps_output)
		# reconstructions,decoder_masked = self.decoder(digitCaps_output)

		return conv_output

	def loss(self,input:torch.Tensor,reconstructions:torch.Tensor,
			 labels:torch.Tensor,predicts:torch.Tensor,size_average=True,m_plus=0.9,m_minus=0.1):
		return self.margin_loss(labels,predicts,size_average,m_plus,m_minus) + \
			   self.reconstruction_loss(input,reconstructions)

	def margin_loss(self,labels:torch.Tensor,predicts:torch.Tensor,
					size_average=True,m_plus=0.9,m_minus=0.1):
		batch_size = predicts.size(0)

		v_k = torch.sqrt((predicts**2).sum(dim=2,keepdim=True))
		plus_item = self.relu(torch.tensor(m_plus,dtype=torch.float32,device=self.device)-v_k).view(batch_size,-1)
		minus_item = self.relu(v_k - torch.tensor(m_minus,dtype=torch.float32,device=self.device)).view(batch_size,-1)

		loss = labels * plus_item**2 + 0.5 * (torch.tensor(1.0,dtype=torch.float32,device=self.device) - labels) * \
				minus_item**2
		loss = loss.sum(dim=1).mean()

		return loss

	def reconstruction_loss(self,input:torch.Tensor,reconstructions:torch.Tensor,size_average=True):
		# print('input')
		# print(input.shape)
		loss = self.mse_loss(reconstructions.view(reconstructions.size(0),-1),input.view(reconstructions.size(0),-1))


		return loss * 0.0005





