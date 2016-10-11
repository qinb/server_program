import sys
import os
import re
import operator

path = 'data/all_stocks_5_minute/'
#path = 'tmp/'
for file in os.listdir(path):
	data_name = file.split('.')
	len_name = len(data_name)
	if((len_name == 1)&((os.path.isdir(path+file))==False)):
		#get the name
		name = re.findall(r'\d+',file)
		print(file)
		#derive the train & val & test' name
		if name != []:
			name = name[0]
			train_file = 'data'+'_'+name+'_'+'train'
			val_file = 'data'+'_'+name+'_'+'val'
			test_file = 'data'+'_'+name+'_'+'test'

		#load data
		dl = open(path+file)
		date_price = []
		for line in dl:
			date_price.append(line.split(',')[::6])
	
		prices = date_price[1::]
		#get 2D list the 0 col
		date = list(zip(*prices)[0])
		date = [(date[i]).split('/')[0] for i in range(len(date))]
		#print date	
		#get the number of years
		year_datanum =[]
		for x in set(date):
			# sort for the first dimension
			year_datanum.append([x,date.count(x)])
		year_datanum.sort(key=lambda ty:ty[0])
		print year_datanum
