# -*- coding: utf-8 -*-
# @Time    : 2020-06-28 10:59
# @Author  : PeterV
# @FileName: predict.py
# @Software: PyCharm

import torch
import numpy as np
from capsNet import CapsNet
from utils import store_results

def predict(capsNet,dataProcessor,num_classes,epoch,device,eval_type,logger):
	capsNet.eval()
	total_predicts = []
	total_labels = []
	if eval_type == 'valid' or eval_type == 'dev':
		dataloader = dataProcessor.valid_loader
	else:
		dataloader = dataProcessor.test_loader
	for batch_id, (test_data, test_target) in enumerate(dataloader):
		test_target = torch.eye(num_classes).index_select(dim=0, index=test_target)

		test_data = test_data.to(device)
		test_target = test_target.to(device)

		# output, reconstructions, masked = capsNet(test_data)
		output = capsNet(test_data)

		# total_predicts.extend(masked.data.cpu().numpy())
		total_predicts.extend(output.data.cpu().numpy())
		total_labels.extend(test_target.data.cpu().numpy())

	scores = sum(np.argmax(np.array(total_labels), 1) == np.argmax(np.array(total_predicts), 1)) / float(len(total_labels))
	logger.info('epoch:{} || {} accuracy:{}'.format(epoch, eval_type,scores))

	return scores,total_predicts,total_labels

def eval(model_path,result_path,dataProcessor,num_classes,epoch,device,eval_type,logger):
	capsNet = CapsNet(num_classes=num_classes,conv_in=3)
	state_dict = torch.load(model_path + '/model_state_dict.pkl')
	capsNet.load_state_dict(state_dict)
	capsNet.to(device)
	valid_scores,valid_predicts,valid_labels = predict(capsNet,dataProcessor,num_classes,epoch,device,'valid',logger)
	test_scores,test_predicts,test_labels = predict(capsNet,dataProcessor,num_classes,epoch,device,'test',logger)
	store_results(test_predicts,test_labels,result_path)
	with open(result_path + '/result.csv','w') as file:
		file.write('valid scores,test scores\n')
		file.write(str(valid_scores)+','+str(test_scores))

	return valid_scores,test_scores

