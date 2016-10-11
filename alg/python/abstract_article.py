import os
import sys
import re
import urllib2


website = 'http://jiqizhixin.com'
path ='./machine_heart'
pathelse ='./machine_heart_index'

modules = ['/edge','/insights','/list/index/id/8','/science','/life','/video','/special']

listed =[]

for m in range(len(modules)):
	for p in range(1,71,1):
		response = urllib2.urlopen(website+modules[m]+'/p/'+str(p))
		html = response.read()
		link_list =re.findall(r"(?<=href=\").+?(?=\")|(?<=href=\').+?(?=\')" ,html)
		for url in link_list:
			data = url.split('/')
			data1 = url.split('#')
			url_section = len(data)
			if url_section >=2:
				if(data[1] == 'article')&(data1[-1]!='comment'):
					listed.append(str(m)+'_'+url)
#collect the different url
linklist=list(set(listed))

#write the file,if the file cann't open,it is writed in path.
#else it is writed in pathelse
for url in linklist:
	url1 = url.split('_')[1]
	c = int(url.split('_')[0])
	url = url1
	dirs = path+modules[c]
		
	if os.path.exists(dirs) ==False:
		os.makedirs(dirs)

	res = urllib2.urlopen(website+url)
	htm = res.read()
	
	try:
		name_utf8 = re.search("<title>.*</title>", htm).group().strip("</title>").split('_')[0]
		tmp = name_utf8.split('/')
		name_utf8 = "_".join(tmp)
		name_utf8 = name_utf8.replace(' ','_')
		try:
			name_unicode = name_utf8.decode('utf-8')
			name_gbk = name_unicode.encode('gbk')
			print name_utf8
			f = open(dirs+'/'+name_gbk+'.html','w')
	#such as FT/ ,those titles can't be normally recongized
	#but <200b>,this situation cann't be omitted
		except:
			f = open(dirs+'/'+name_utf8+'.html','w')
		f.write(htm)	
		f.close()
	except:
		name = url.split('/')[2]
		f = open(pathelse+'/'+name+'.html','w')
		f.write(htm)	
		f.close()

