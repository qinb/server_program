import sys
import os

interval = [0,1,3]

for i in range(4):
	for j in range(3):

#i =(int)(sys.argv[1])
#command3 = "theano_flags=device=gpu%d python lr_return.py %d | tee -a ./log/lr_hour_log_%d" %(i,i,i)
#os.system(command3)

#command4 = "theano_flags=device=gpu%d python lr_return_v1.py %d | tee -a ./log/lr_hour_log_v1_%d" %(i,i,i)
#os.system(command4)

	#command2 = "THEANO_FLAGS=device=gpu%d python lstm_return.py %d | tee -a ./log/lstm_hour_log_%d" %(i,i,i)
		command2 = "python lstm_bin.py %d %d | tee -a ./log/lstm_%ddays_log_%d" %(i,interval[j],interval[j],i)
		os.system(command2)




