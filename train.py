# -*- coding: utf-8 -*-
# @Time    : 2020-06-25 14:40
# @Author  : PeterV
# @FileName: train.py
# @Software: PyCharm
import numpy as np

import os
import torch
import logging
import random
from capsNet import CapsNet
from tqdm import tqdm
from predict import predict,eval
from dataProcessor import DataProcessor
from config import Config

index = int(random.random() * 10000)
# config and log
cfg = Config.get_args()
model_path = os.path.join(cfg.model_path,cfg.version,str(index))
result_path = os.path.join(cfg.result_path,cfg.version,str(index))
if not os.path.exists(model_path):
	os.makedirs(model_path)
if not os.path.exists(result_path):
	os.makedirs(result_path)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
log_dir = os.path.join(cfg.log_dir,cfg.version)
if not os.path.exists(log_dir):
	os.makedirs(log_dir)
file_handler = logging.FileHandler(os.path.join(log_dir,str(index)+'.log'))
logger.addHandler(file_handler)

batch_size = cfg.batch_size
epochs = cfg.epochs
dataset = cfg.dataset
if dataset=='MNIST':
	num_classes = 10
else:
	num_classes = 26

dataProcessor = DataProcessor(batch_size=batch_size,dataset=dataset,data_path='data/char74k_preprocessed')
if torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')

capsNet =  CapsNet(conv_in=3,num_classes=num_classes).to(device)
cseloss = torch.nn.CrossEntropyLoss()
adamOptimizer = torch.optim.Adam(params=capsNet.parameters())
best_val_acc = -1
best_val_epoch = 0
for epoch in tqdm(range(epochs)):
	epoch_loss = 0
	for batch_id, (data,target) in enumerate(dataProcessor.train_loader):
		capsNet.train()
		target = torch.eye(num_classes).index_select(dim=0,index=target)

		data = data.to(device)
		target = target.to(device)

		adamOptimizer.zero_grad()
		# output,reconstructions,masked = capsNet(data)
		output = capsNet(data)

		# print(output.shape)
		# loss = capsNet.loss(data,reconstructions,target,output)
		loss = cseloss(output,target.argmax(dim=-1))
		loss.backward()
		adamOptimizer.step()
		epoch_loss += loss

		if batch_id % 5 == 0:
			logger.info('epoch:{} || batch id:{} || train loss:{}'.format(epoch
													,batch_id,loss.data))
			break

		del data
		del target
	logger.info('epoch:{} || epoch loss:{} || best epoch:{} || best val acc: {}'.format(
			epoch,epoch_loss/float(len(dataProcessor.train_loader)),best_val_epoch,best_val_acc))

	capsNet.eval()
	scores,valid_preds,valid_labels = predict(capsNet, dataProcessor, num_classes, epoch, device, 'valid',logger)
	scores_test,test_preds,test_labels = predict(capsNet,dataProcessor,num_classes,epoch,device,'test',logger)
	if scores_test >= best_val_acc:
		best_val_acc = scores_test
		best_val_epoch = epoch
		torch.save(capsNet.state_dict(),model_path + '/model_state_dict.pkl')
	elif epoch - best_val_epoch > cfg.patience:
		logger.info("Since the val acc has not improved after {} epoch(s), "
					"you have got an excellent enough model,congratulations!".format(cfg.patience))
		break

valid_scores,test_scores = eval(model_path, result_path, dataProcessor, num_classes, best_val_epoch, device, 'eval',logger)
logger.info("Valid Acc:{} || Test Acc:{}".format(valid_scores,test_scores))

