#########################################################################
# File Name: rnn.py
# Author: james
# mail: zxiaoci@mail.ustc.edu.cn
#########################################################################

from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy as np

import theano
import theano.tensor as T
import lasagne

#SEQ_LENGTH = 7
#N_HIDDEN = 200
LEARNING_RATE = 0.01
GRAD_CLIP = 100
NUM_EPOCHS = 100
BATCH_SIZE = 1
INPUT_SIZE = 4
max_grad_norm = 10

axis =(int)(sys.argv[1])
SEQ_LENGTH =(int)(sys.argv[2])
N_HIDDEN = (int)(sys.argv[3])

def load_data(dataset):
	print('... loading data')
	data = open(dataset, 'r')
	prices = []
	for line in data:
		prices.append(line.split('\n')[0].split('\t')[1:5])
	return prices[1:]

print("Loading data ...")
train_data = load_data('data/ifly_train')
train_data_size = len(train_data) 
test_data = load_data('data/ifly_test')
test_data_size = len(test_data)
val_data = load_data('data/ifly_val')
val_data_size = len(val_data) 


def minmax_norm(data):
	max = np.max(data)
	min = np.min(data)
	return (data - min) / (max - min)



def gen_data(p, data, batch_size = BATCH_SIZE, input_size = INPUT_SIZE):
	x = np.zeros((batch_size, SEQ_LENGTH, input_size))
	y = np.zeros((batch_size,2))
	y_old = np.zeros(batch_size)

	for n in range(batch_size):
		ptr = n
		for i in range(SEQ_LENGTH):
			for j in range(0, input_size):
				x[n, i, j] = data[p+ptr+i][j]
		#y[n] = data[p+ptr+SEQ_LENGTH][3]
		next_y = data[p+ptr+SEQ_LENGTH][axis]
		y_old[n] = data[p+ptr+SEQ_LENGTH-1][axis]
		if((float)(next_y) >= (float)(y_old[n])):
			y[n] = [1, 0]
		else:
			y[n] = [0, 1]
	
	# Center the inputs and outputs
	#x -= x.reshape(-1,input_size).mean(axis=0)
	#y_mean = y.mean()
	x=minmax_norm(x)
	#??
	#y -= y_old
   #	y = (y - y_old)*10 / y_old
	
	return (x.astype(theano.config.floatX), y.astype(theano.config.floatX),
			y_old.astype(theano.config.floatX))



class MLP(object):

	def __init__(self,shape,n_hidden):
		self.shape = shape
		self.n_hidden = n_hidden

		self.l_in = lasagne.layers.InputLayer(shape)
		self.l_forward = lasagne.layers.DenseLayer(self.l_in,num_units=self.n_hidden,nonlinearity=lasagne.nonlinearities.tanh)
		self.l_out = lasagne.layers.DenseLayer(self.l_forward,num_units=2,nonlinearity=lasagne.nonlinearities.softmax)
	

if __name__ == '__main__':

	if(len(sys.argv)<=3):
		print("use:python lstm_bin_test.py axis SEQ_LENGTH N_HIDDEN")
		sys.exit()	
	print("Building network ...")

	shape = (BATCH_SIZE,SEQ_LENGTH,INPUT_SIZE) 
	classifier = MLP(shape,N_HIDDEN)

	target_values = T.matrix('target_output')
	
	network_output = lasagne.layers.get_output(classifier.l_out)
	predicted_values = network_output

	#cost = T.mean((predicted_values - target_values)**2)
	#cost = T.mean(np.sum([(predicted_values[i]-target_values[i])**2 for i in len(target_values)]))
	cost = lasagne.objectives.categorical_crossentropy(predicted_values, target_values)
	cost = cost.mean()

	accuracy = T.mean(T.eq(T.argmax(predicted_values,axis=1),T.argmax(target_values,axis=1)),dtype=theano.config.floatX)

	all_params = lasagne.layers.get_all_params(classifier.l_out,trainable=True)

	print("Computing updates ...")
	all_grads = T.grad(cost,all_params)
	all_grads = [T.clip(g,-5,5) for g in all_grads]

	all_grads,norm = lasagne.updates.total_norm_constraint(all_grads,max_grad_norm,return_norm=True)

	sh_lr = theano.shared(lasagne.utils.floatX(LEARNING_RATE))

	updates = lasagne.updates.adagrad(cost, all_params, learning_rate=sh_lr)

	print("Compiling functions ...")
	train = theano.function([classifier.l_in.input_var, target_values], 
			[cost, predicted_values], updates=updates, allow_input_downcast=True)
	compute_cost = theano.function([classifier.l_in.input_var, target_values], 
			#[cost, predicted_values, test_acc], allow_input_downcast=True)
			[cost, predicted_values,accuracy], allow_input_downcast=True)

	print("Training ...")
	ftrain = open('train_mlp_bin.txt', 'wb')
	fval = open('val_mlp_bin.txt', 'wb')
	ftest = open('test_mlp_bin.txt', 'wb')
	p = 0
	last_cost_val = 10000
	best_cost_val = 10000
	try:
		step = SEQ_LENGTH + BATCH_SIZE - 1
		for it in xrange(NUM_EPOCHS):
			print('epoch %d, lrate = %f' % (it, sh_lr.get_value()))
			for _ in range(train_data_size / BATCH_SIZE):
				x,y,y_mean = gen_data(p, train_data)
				cost, pred = train(x,y)
			#	pred = T.argmax(pred,axis=1)
			#	y = T.argmax(y,axis=1)
				ftrain.write("predicted = {}, actual_value = {}\n"
						.format((pred), (y)))
				# to reuse previous batch, i.e. last batch is data[0:10], next batch 
				# will become data[1:11] instead of data[11:20]

				p += BATCH_SIZE
				if(p + step >= train_data_size):
					#print('Carriage return')
					p = 0

			pp = 0
			cost_val = 0
			#for _ in range(val_data_size / SEQ_LENGTH / BATCH_SIZE):
			acc_val = 0

			n_iter = (val_data_size - step) / BATCH_SIZE
			for _ in range(n_iter):
				x,y,y_mean = gen_data(pp, val_data)
				cost, pred, acc = compute_cost(x, y)
				fval.write("predicted = {}, actual_value = {}\n"
						.format((pred), (y)))
				cost_val += cost
				acc_val += acc
				pp += BATCH_SIZE
				if(pp + step >= val_data_size):
					break
			cost_val /= n_iter
			acc_val /= n_iter
			print("cost = {}, acc = {}".format(cost_val, acc_val))

			# halve lrate
			if (last_cost_val <= cost_val * 1.001):
				lasagne.layers.set_all_param_values(classifier.l_out, all_param_values)
				current_lr = sh_lr.get_value()
				sh_lr.set_value(lasagne.utils.floatX(current_lr / 2))
				if (sh_lr.get_value() <= 10e-5):
					break
			else:
				all_param_values = lasagne.layers.get_all_param_values(classifier.l_out)
				best_cost_val = cost_val
			last_cost_val = cost_val

		lasagne.layers.set_all_param_values(classifier.l_out, all_param_values)
		pp = 0
		acc_test = 0
		n_iter = (test_data_size - step) / BATCH_SIZE
		for _ in range(n_iter):
			x,y,y_mean = gen_data(pp, test_data)
			cost, pred, acc = compute_cost(x, y)
			#pred = T.argmax(pred,axis=1)
			#y = T.argmax(y,axis=1)
			ftest.write("predicted = {}, actual_value = {}\n"
					.format((pred), (y)))
			acc_test += acc
			pp += BATCH_SIZE
			if(pp + step >= test_data_size):
				break
		acc_test /= n_iter
		print("test acc = {}".format(acc_test))
	except KeyboardInterrupt:
		pass
