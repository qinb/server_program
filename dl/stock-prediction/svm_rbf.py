import sys
import os
import numpy as np

import theano
import theano.tensor as T
import lasagne
import math
import math.e as e
from lr import load_data

BATCH = 1
SEQ_LEN = 1
HOUR = 12
INPUT = 4 
e = 0.03
# sample' row
row = len(data) - seq - HOUR
# sample' column
col = INPUT*seq
axis =(int)(sys.argv[1])

train_file = 'data/ifly_train'
valid_file = 'data/ifly_valid'
test_file = 'data/ifly_test'

train_data = load_data(train_file)
train_size = len(train_data)
test_data = load_data(test_file)
test_size = len(test_data)
valid_data = load_data(valid_file)
valid_size = len(valid_data)

#E store all samples' error
# g stores all samples' result
E = np.zeros(row)
g = np.zeros(row)


def gen_sample(data,seq = SEQ_LEN):
	
	all_sample = np.zeros([row,col+1])
	for i in range(row):
		for j in range(col):
			all_sample[i][j] = ((math.log((float)(data[i+(j/INPUT)][j%INPUT])) - math.log((float)(data[i+(j/INPUT)+HOUR][j%INPUT])))/math.log(2)
		values = (math.log((float)(data[i+(col/INPUT)][axis])) - math.log((float)(data[i+(col/INPUT)+HOUR][axis])))/math.log(2)
		if values >= 0:
			all_sample[i][col] = 1
		else:
			all_sample[i][col] = 0
	
	return all_sample


class svm(object):
	# N is all samples' nums
	# the following data is dealed by gen_sample 
	def __init__(self,C,N):
		self.C = C
		self.N = N
	
	
	def rbf(xj,xi,widparas):
		square_sum = sum([(xj[i]-xi[i])**2 for i in range(len(xj))])
		return power(e,square_sum)/(2*(widparas**2))	


	# the following data is dealed by gen_sample 
	def g_fun(data,alpha,b):
		tmp = 0
		for t in range(len(data)):
			for i in range(len(data)):
				if i!=t:
					tmp = tmp + alpha[i]*data[i][col]*rbf(data[i],data[t],wid)
			g[t] = tmp + b
		
		

	# t is data index
	def cal_E(data,alpha,b):
		
		g_fun(data,alpha,b)
		for i in range(self.N):
			E[i] = g[i] -data[col]	
		return E	


	def seq_minnimal_opt(data):
		b = 0
		alpha = np.zeros(self.N,dtype=theano.config.floatX)
		for i in range(self.N):
			cal_E(data,alpha,b)
			if (alpha[i] > 0) && (alpha[i] < self.C):
				if !((1-e <= data[i][col]*g[i])&&(data[i][col]*g[i]<=1+e)):
					flag1 = i
			if (alpha[i] == 0):
				if !(data[i][col]*g[i] >= 1+e):
					flag1 = i

			if (alpha[i])

					

if __name__ == '__main__':
	gen_sample(train_data)
