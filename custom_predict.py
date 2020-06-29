# -*- coding: utf-8 -*-
# @Time    : 2020-06-29 17:05
# @Author  : PeterV
# @FileName: custom_predict.py
# @Software: PyCharm

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from capsNet import CapsNet
from utils import char74k_id2labels


pic_path = ''
dataset = 'character'

if torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')

def predict_picture(pic_path,model_path,dataset,device):
	if dataset == 'char74k':
		size = 20
		num_classes = 26
	else:
		size = 28
		num_classes = 10
	transform = transforms.Compose([
		transforms.Scale(size),
		transforms.CenterCrop((size,size)),
		transforms.ToTensor(),
		transforms.Normalize((0.1307,),(0.3081,))
	])

	img = Image.open(pic_path)
	img_tensor = transform(img)
	img_tensor = img_tensor.unsqueeze(0)
	img_tensor = img_tensor.to(device)

	model = CapsNet(num_classes=num_classes,conv_in=3)
	model = model.to(device)
	model.eval()
	state_dict = torch.load(model_path)
	model.load_state_dict(state_dict)

	score = model(img_tensor)
	probs = nn.functional.softmax(score,dim=-1)
	max_value,index = torch.max(probs,dim=-1)
	print('The picture {} is:{}'.format(pic_path.split('/')[-1],char74k_id2labels[int(index)]))

if __name__ == '__main__':
	pic_path = 'data/custom/1.jpg'
	model_path = 'result/alexnet_debug/9201/model_state_dict.pkl'
	predict_picture(pic_path,model_path,'char74k',device)
