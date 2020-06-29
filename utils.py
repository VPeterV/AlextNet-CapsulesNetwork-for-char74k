# -*- coding: utf-8 -*-
# @Time    : 2020-06-29 15:53
# @Author  : PeterV
# @FileName: utils.py
# @Software: PyCharm

import os
import numpy as np

char74k_id2labels={
	item:chr(item+97) for item in range(0,26)
}

def store_results(predicts,labels,result_path):
	file_name = result_path + '/case.csv'
	if not os.path.exists(file_name):
		with open(file_name,'w') as file:
			file.write('predicts,labels\n')
	with open(file_name,'a') as file:
		for pred,label in zip(predicts,labels):
			file.write(str(char74k_id2labels[np.argmax(pred)]) + ',' + str(char74k_id2labels[np.argmax(label)]) + '\n')

