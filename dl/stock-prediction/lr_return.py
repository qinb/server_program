import os
import sys
import numpy as np
import theano
import theano.tensor as T
import six.moves.cPickle as pickle
import lasagne
import math
import re

SEQ = 8     # how many steps to unroll
LEARNING_RATE = 0.01   # learning rate
GRAD_CLIP = 100        # 
#NUM_EPOCHS = 100
NUM_EPOCHS = 50
BATCH_SIZE = 20
INPUT_SIZE = 4
INI = lasagne.init.Uniform(0.1)
max_grad_norm = 10
interval = 12

axis = (int)(sys.argv[1])
flag = (int)(sys.argv[2])

train_log =[('result/train_lr_hour_%d.txt' %axis),('result/train_lr_hour_v1_%d.txt' %axis)]
val_log =[('result/train_lr_hour_%d.txt' %axis),('result/train_lr_hour_v1_%d.txt' %axis)]
test_log =[('result/train_lr_hour_%d.txt' %axis),('result/train_lr_hour_v1_%d.txt' %axis)]
model_file = [('model/model_lr_%d' %axis),('model/model_lr_v1_%d' %axis)]


#model_file =('model/model_lr_%d' %axis)


train = 'data/data_000002_train'
val   = 'data/data_000002_val'
test  = 'data/data_000002_test'

#train = 'data/data_002185_train'
#val   = 'data/data_002185_val'
#test  = 'data/data_002185_test'

def clean_data(data):
	k = 0
	cleaned_data = np.zeros([len(data),INPUT_SIZE],dtype=np.float)
	for i in range(len(data)-1):
		if data[i] != data[i+1]:
			cleaned_data[k] = data[i]	
			k = k + 1

	if data[len(data)-2]!= data[len(data)-1]:
		cleaned_data[k] = data[len(data)-1]	
	
	return cleaned_data[0:(k+1)]


def load_data(dataset):
	print('... loading data')
	data = open(dataset, 'r')
	prices = []
	for line in data:
		prices.append(line.split('\n')[0].split('\t')[2:6])
		#prices.append(line.split('\n')[0].split('\t')[1:5])
	prices = clean_data(prices[1::])
	return prices[1::]


print("Loading data ...")
train_data = load_data(train)
train_size = len(train_data) 
val_data = load_data(val)
val_size = len(val_data) 
test_data = load_data(test)
test_size = len(test_data) 

def gen_data(data, ptr,batch = BATCH_SIZE, input_size = INPUT_SIZE):
	x = np.zeros((batch, SEQ, input_size))
	y = np.zeros((batch,2))

	for b in range(batch):
		for s in range(SEQ):
			for i in range(input_size):
				x[b, s, i] = (math.log((float)(data[ptr+b+s][i]))-math.log((float)(data[b+ptr+s+interval][i])))/math.log(2)
		#temp = (math.log((float)(data[b+ptr+SEQ][axis]))-math.log((float)(data[b+ptr+SEQ+interval][axis])))/math.log(2)
		if flag ==0:
			temp = (math.log((float)(data[b+ptr+SEQ][axis]))-math.log((float)(data[b+ptr+SEQ+interval][axis])))/math.log(2)
		if flag ==1:
			temp = (math.log((float)(data[p+ptr+SEQ_LENGTH+1][axis]))-math.log((float)(data[p+ptr+SEQ_LENGTH+interval+1][axis])))/math.log(2)
		if ((float)(temp) <= 0):
			y[b] = [1, 0]
		else:
			y[b] = [0, 1]
	return (x.astype(theano.config.floatX), y.astype(theano.config.floatX))

if __name__ == '__main__':
	
	print("Building network ...")
	l_in = lasagne.layers.InputLayer(shape=(BATCH_SIZE, SEQ, INPUT_SIZE))
	l_out = lasagne.layers.DenseLayer(l_in, num_units=2, 
			nonlinearity=lasagne.nonlinearities.softmax)
	
	target_values = T.matrix('target_output')
	
	network_output = lasagne.layers.get_output(l_out)
	predicted_values = network_output
	cost = lasagne.objectives.categorical_crossentropy(predicted_values, target_values)
	cost = cost.mean()

	test_acc = T.mean(T.eq(T.argmax(predicted_values, axis=1),
		T.argmax(target_values, axis=1)), dtype=theano.config.floatX)

	all_params = lasagne.layers.get_all_params(l_out, trainable=True)

	print("Computing updates ...")
	all_grads = T.grad(cost, all_params)
	all_grads = [T.clip(g, -5, 5) for g in all_grads]
	all_grads, norm = lasagne.updates.total_norm_constraint(
			all_grads, max_grad_norm, return_norm=True)

	sh_lr = theano.shared(lasagne.utils.floatX(LEARNING_RATE))

	updates = lasagne.updates.adagrad(cost, all_params, learning_rate=sh_lr)

	print("Compiling functions ...")
	train = theano.function([l_in.input_var, target_values], 
			[cost, predicted_values], updates=updates, allow_input_downcast=True)
	compute_cost = theano.function([l_in.input_var, target_values], 
			[cost, predicted_values, test_acc], allow_input_downcast=True)

	print("Training ...")
	ftrain = open(train_log[flag], 'wb')
	fval = open(val_log[flag], 'wb')
	ftest = open(test_log[flag], 'wb')
	
	ptr = 0
	last_cost_val = 10000
	best_cost_val = 10000
	try:
		step = SEQ + interval + BATCH_SIZE - 1
		for it in xrange(NUM_EPOCHS):
			print('epoch %d, lrate = %f' % (it, sh_lr.get_value()))
			for _ in range(train_size / BATCH_SIZE):
				x,y = gen_data(train_data,ptr)
				cost, pred = train(x,y)
			#	tmp = x[0]
		#	axis_data = [S[axis] for S in tmp]	
				ftrain.write("samples = {},predicted = {}, actual_value = {}\n"
						.format((x),(pred),(y)))
				ptr += BATCH_SIZE
				#print(ptr)
				if(ptr + step >= train_size):
					ptr = 0

			pp = 0
			cost_val = 0
			acc_val = 0
			n_iter = (val_size - step) / BATCH_SIZE
			for _ in range(n_iter):
				x,y = gen_data(val_data,pp)
				cost, pred, acc = compute_cost(x, y)
		#		tmp = x[0]
		#		axis_data = [S[axis] for S in tmp]	
				fval.write("samples ={}, predicted = {}, actual_value = {}\n"
						.format((x),(pred), (y)))
				cost_val += cost
				acc_val += acc
				pp += BATCH_SIZE
				if(pp + step >= val_size):
					break
			cost_val /= n_iter
			acc_val /= n_iter
			print("cost = {}, acc = {}".format(cost_val, acc_val))

			# halve lrate
			if (last_cost_val <= cost_val * 1.005):
				lasagne.layers.set_all_param_values(l_out, all_param_values)
				current_lr = sh_lr.get_value()
				sh_lr.set_value(lasagne.utils.floatX(current_lr / 2))
				if (sh_lr.get_value() <= 10e-4):
					break
			else:
				all_param_values = lasagne.layers.get_all_param_values(l_out)
				best_cost_val = cost_val
			last_cost_val = cost_val

		model = open(model_file[flag],'w')	
		pickle.dump(all_param_values,model)

		lasagne.layers.set_all_param_values(l_out, all_param_values)
		pp = 0
		acc_test = 0
		n_iter = (test_size - step) / BATCH_SIZE
		for _ in range(n_iter):
			x,y = gen_data(test_data,pp)
			cost, pred, acc = compute_cost(x, y)
			ftest.write("samples = {}, predicted = {}, actual_value = {}\n"
					.format((x),(pred), (y)))
			acc_test += acc
			pp += BATCH_SIZE
			if(pp + step >= test_size):
				break
		acc_test /= n_iter
		ftrain.close()
		fval.close()
		ftest.close()
		#os.rename('result/train_lr_hour.txt',
		print("test acc = {}".format(acc_test))
	except KeyboardInterrupt:
		pass
