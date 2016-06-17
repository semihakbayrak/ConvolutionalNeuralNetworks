#-*- coding: utf-8 -*- 

#BoW.py
#Naive Bayes for comparison

import os
import re
import locale
from collections import Counter
import numpy as np

#directories for training and test
directory_train = '/Users/semihakbayrak/ConvNet4/training'
directory_test = '/Users/semihakbayrak/ConvNet4/test'

#categoriess
categorylist = []
category_vocabularies = {}
for filename in os.listdir(directory_train):
	categorylist.append(filename)
for category in categorylist:
	if category == '.DS_Store':
		categorylist.remove('.DS_Store')

priors = np.array([]) 
conclistall = []
for category in categorylist:
	directory_category = directory_train+'/'+str(category)
	textlist = []
	
	for filename in os.listdir(directory_category):
		textlist.append(filename)
	
	for text in textlist:
		if text == '.DS_Store':
			textlist.remove('.DS_Store')
	
	priors = np.append(priors,1.0*len(textlist)) #prior probabilities for categories
	conclist1 = []
	
	#tokenization and training
	for text in textlist:
		directory_text = directory_category+'/'+str(text)
		textfile = open(directory_text,'r').read()
		tok = textfile.decode('utf-8','ignore')
		tok2 = tok.lower()
		tok3 = re.sub(r"\d+",'',tok2,flags=re.U)    #remove numbers
		tok4 = re.sub(r"\W+",'\n',tok3,flags=re.U)  #remove non alphanumerics with new line
		conclist2 = tok4.split()
		conclist1 = conclist1 + conclist2 #all the words used by category in a list

	category_vocabularies[str(category)]=Counter(conclist1) #word frequencies for categories in dictionaries
	conclistall = conclistall + conclist1 #all the words used by all categories in a list

vocabulary = Counter(conclistall) #vocabulary for words with their frequencies in a dictionary
priors = priors/sum(priors) #prior normalization

#number of words used by categories
wordcountscategory = []
for i in range(len(categorylist)):
	wordcountscategory.append(sum(category_vocabularies[str(categorylist[i])].values()))

counttrue = 0
countfalse = 0

for category in categorylist:
	directory_category = directory_test+'/'+str(category)
	textlist = []
	for filename in os.listdir(directory_category):
		textlist.append(filename)
	for text in textlist:
		if text == '.DS_Store':
			textlist.remove('.DS_Store')
	for text in textlist:
		#tokenization for test inputs
		directory_text = directory_category+'/'+str(text)
		textfile = open(directory_text,'r').read()
		tok = textfile.decode('utf-8','ignore')
		tok2 = tok.lower()
		tok3 = re.sub(r"\d+",'',tok2,flags=re.U)
		tok4 = re.sub(r"\W+",'\n',tok3,flags=re.U)
		tlist = tok4.split()
		#multinomial naive bayes with laplace smoothing
		logprob = np.log2(priors)
		alfa = 0.01 
		for word in tlist:
			for i in range(len(categorylist)):
				if word in category_vocabularies[str(categorylist[i])]:
					probw = 1.0*(category_vocabularies[str(categorylist[i])][word]+alfa)/(wordcountscategory[i]+alfa*len(vocabulary))
				else:
					probw = 1.0*alfa/(wordcountscategory[i]+alfa*len(vocabulary))
				logprob[i] = logprob[i] + np.log2(probw)
		print categorylist[np.argmax(logprob)]
		#count correct and incorrect instances
		if categorylist[np.argmax(logprob)]==category:
			print '1'
			counttrue = counttrue + 1
		else:
			print '0'
			countfalse = countfalse + 1
		
		

accuracy = 1.0*counttrue/(counttrue+countfalse)
print accuracy


