# -*- coding: utf-8 -*-
# @Time    : 2020-06-29 9:35
# @Author  : PeterV
# @FileName: config.py
# @Software: PyCharm

import argparse

class Config:
	@staticmethod
	def get_args():
		parser = argparse.ArgumentParser()

		# parameters
		parser.add_argument('--version',default='alexnet')
		parser.add_argument('--dataset',default='char74k',choices=['MNIST','char74k'])
		parser.add_argument('--num_classes',default=26,type=int)
		parser.add_argument('--data_path',default='./data/char74k_preprocessed')
		parser.add_argument('--model_path',default='./result')
		parser.add_argument('--log_dir',default='./log')
		parser.add_argument('--result_path',default='./result')
		parser.add_argument('--custom_pic',default='./data/custom')

		# hyper-parameters
		parser.add_argument('--epochs',default=10000,type=int)
		parser.add_argument('--batch_size',default=64,type=int)
		parser.add_argument('--patience',default=1000,type=int)

		config = parser.parse_args()

		return config
