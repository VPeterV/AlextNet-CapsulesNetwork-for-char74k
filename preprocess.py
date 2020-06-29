# -*- coding: utf-8 -*-
# @Time    : 2020-06-28 13:46
# @Author  : PeterV
# @FileName: preprocess.py
# @Software: PyCharm

import os
import random
import shutil

def split_files(file_path,result_path,ratio=0.1):
	files_name = os.listdir(file_path)
	file_len = len(files_name)
	# print(file_len)
	random.shuffle(files_name)
	train_index = int(file_len*(1-ratio*2))
	# print(train_index)
	train_data = files_name[:train_index]
	valid_data = files_name[train_index:int(train_index+file_len*ratio)]
	test_data = files_name[int(train_index+file_len*ratio):]
	label = file_path.split('\\')[-1]
	# print(label)
	makepath(result_path,label)
	for item in train_data:
		shutil.copy(os.path.join(file_path,item),os.path.join(result_path,'train',label,item))
	for item in valid_data:
		shutil.copy(os.path.join(file_path,item),os.path.join(result_path,'dev',label,item))
	for item in test_data:
		shutil.copy(os.path.join(file_path,item),os.path.join(result_path,'test',label,item))

def makepath(result_path,label):
	if not os.path.exists(result_path):
		os.makedirs(result_path)
	if not os.path.exists(os.path.join(result_path,'train',label)):
		os.makedirs(os.path.join(result_path,'train',label))
	if not os.path.exists(os.path.join(result_path, 'dev',label)):
		os.makedirs(os.path.join(result_path, 'dev',label))
	if not os.path.exists(os.path.join(result_path, 'test',label)):
		os.makedirs(os.path.join(result_path, 'test',label))

if __name__ == '__main__':
	file_path = 'data/char74k'
	result_path = 'data/chat74k_preprocessed'
	all_labels = os.listdir(file_path)
	for label in all_labels:
		# print(label)
		# print(os.path.join(file_path,label))
		split_files(os.path.join(file_path,label),result_path)
