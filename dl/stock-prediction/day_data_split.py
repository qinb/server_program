################################################################
#File Name:day_data_split.py
#Author:boqin
#mail:qinb@mail.ustc.edu.cn
################################################################


import sys
import os
import re
import operator

path = './'
for file in os.listdir(path):
	data_name = file.split('.')
	#len_name = len(data_name)
	if((data_name[-1]=='txt')&((os.path.isdir(path+file))==False)):
		#get the name
		name = re.findall(r'\d+',file)
		print(file)
		#derive the train & val & test' name
		if name != []:
			name = name[0]
			train_file = 'data_splited/'+'data'+'_'+name+'_'+'train'
			val_file = 'data_splited/'+'data'+'_'+name+'_'+'val'
			test_file = 'data_splited/'+'data'+'_'+name+'_'+'test'

			#load data
			dl = open(path+file)
			num = 0
			ftrain = open(train_file,'w')
			fval = open(val_file,'w')
			ftest = open(test_file,'w')
			data_size = 0
			
			for line in dl:
				data_size = data_size + 1
			dl.close()	
			train_splitpoint = (int)(data_size*0.6)
			valid_splitpoint = (int)(data_size*0.2) + train_splitpoint
			
			dl =open(path+file)
			for line in dl:
				if (num >=0)&(num<=train_splitpoint):
					ftrain.write(line)
				elif (num>train_splitpoint)&(num<=valid_splitpoint):
					fval.write(line)
				else:
					ftest.write(line)
				num = num + 1

			ftrain.close()
			fval.close()
			ftest.close()
			dl.close()
