#-*- coding: utf-8 -*- 
#word2vec.py
#this script uses gensim library to implement word2vec algorithm
import os
import re
import locale
from collections import Counter
import numpy as np
import gensim

#iteration class which will be used to train word2vec algorithm. Returns sentences as a list of words
class SentenceIterator:
	def __init__(self,directory,sw):
		self.directory = directory
		self.stop_words = sw
		self.categorylist = []
		for filename in os.listdir(self.directory):
			self.categorylist.append(filename)
		for category in self.categorylist:
			if category == '.DS_Store':
				self.categorylist.remove('.DS_Store')
	def __iter__(self):
		for category in self.categorylist:
			directory_category = self.directory+'/'+str(category)
			for textname in os.listdir(directory_category):
				directory_text = directory_category+'/'+str(textname)
				textfile = open(directory_text,'r').read()
				textfile = textfile.decode('utf-8','ignore')
				doclist = [ line for line in textfile ]
				docstr = '' . join(doclist)
				sentences = re.split(r'[.!?]', docstr)
				for s in range(len(sentences)):
					sentence = sentences[s]
					sentence = sentence.lower()
					sentence = re.sub(r"\d+",'',sentence,flags=re.U)    #remove numbers	
					sentence = re.sub(r"\W+",'\n',sentence,flags=re.U)  #remove non alphanumerics with new line
					sentence = sentence.split()
					words = [w for w in sentence if w not in self.stop_words] 
					sentence = ' '.join(words)
					wordlist = sentence.split()
					yield wordlist



fileNameSW = '/Users/semihakbayrak/ConvNet4/turkish_stopwords.txt'

textfile = open(fileNameSW,'r').read()
textfile = textfile.decode('utf-8')
textfile = textfile.split()
stop_words = [w for w in textfile]


direc = '/Users/semihakbayrak/ConvNet4/42bin_haber/news'
sentences = SentenceIterator(direc,stop_words)
model = gensim.models.Word2Vec(sentences, size=50, window=4, min_count=5)
model.save("/Users/semihakbayrak/ConvNet4/42bin_haber_w2v_2")

