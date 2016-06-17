#this code creates training and test folders
#training files contain 60percent of the corresponding texts of category
#text files contain the remaining 40 percent 
#but in CNN classification task, I didnt able to use all of them due to high computation time
import os
from shutil import copyfile
from random import shuffle


directory = '/Users/semihakbayrak/ConvNet4/42bin_haber/news'

categorylist = []
for filename in os.listdir(directory):
	categorylist.append(filename)
for category in categorylist:
	if category == '.DS_Store':
		categorylist.remove('.DS_Store')


#creating new directories for training and test
if os.path.isdir('./training'):
	pass
else:
	os.mkdir('./training')

if os.path.isdir('./test'):
	pass
else:
	os.mkdir('./test')

#creating category folders in the training and test folders
for category in categorylist:
	if os.path.isdir('./training/'+str(category)):
		pass
	else:
		os.mkdir('./training/'+str(category))
	if os.path.isdir('./test/'+str(category)):
		pass
	else:
		os.mkdir('./test/'+str(category))

generalpath = os.getcwd()

#copying category texts randomly to new training and test folders
for category in categorylist:
	directory_category = directory+'/'+str(category)
	textlist = []
	for filename in os.listdir(directory_category):
		textlist.append(filename)
	for text in textlist:
		if text == '.DS_Store':
			textlist.remove('.DS_Store')
	shuffle(textlist)
	count60perc = int(len(textlist)*0.6)
	count = 0
	for text in textlist:
		src = directory_category+'/'+str(text)
		if count < count60perc:
			dst = generalpath+'/training/'+str(category)+'/'+str(text)
		else:
			dst = generalpath+'/test/'+str(category)+'/'+str(text)
		copyfile(src,dst)
		count = count + 1

