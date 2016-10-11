#####################################################
#File Name:hyper_select.py
#Author:boqin
#mail:qinb@mail.ustc.edu.cn
######################################################
import os
import sys

for j in range(100,250,20):
	for i in range(6,8,1):
		k = 2*i+6
		command = "THEANO_FLAGS=device=gpu%d python lstm_3paras.py 1 %d %d | tee -a ./log/lstm_log_%d_%d" %(4,k,j,k,j)
		os.system(command)
		os.system("THEANO_FLAGS=device=gpu%d python mlp_3paras.py 1 %d %d | tee -a ./log/mlp_log_%d_%d" %(4,k,j,k,j))
	


#hyper_select
listfile=os.listdir("log")
acc = []
fmlp = open('mlp_log','w')
flstm = open('lstm_log','w')

for line in listfile:
	print line
	fl = open('log/'+line,'r')
#	if len(fl)!=0:
	target =""
	mline = fl.readlines()
	target = mline[-1].split('=')[1]
	print target

	res = line.split('_')
	if res[0] == "lstm":
		flstm.write("combine= {}, acc = {}\n"
				.format((res[2:4]), (target)))
	else:
		fmlp.write("combine= {}, acc = {}\n"
				.format((res[2:4]), (target)))
		
