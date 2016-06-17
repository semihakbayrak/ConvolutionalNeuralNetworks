#-*- coding: utf-8 -*- 

#convTest.py
#test part of CNN

from inputCreator3 import InOut
import os
import re
import locale
from collections import Counter
import numpy as np
import gensim

directory = '/Users/semihakbayrak/ConvNet4'

categories = ["dunya","ekonomi","genel","guncel","kultur-sanat","magazin","planet","saglik","siyaset","spor","teknoloji","turkiye","yasam"]
weightDict = {}

#Stop words
fileNameSW = '/Users/semihakbayrak/ConvNet4/turkish_stopwords.txt'
textfile = open(fileNameSW,'r').read()
textfile = textfile.decode('utf-8')
textfile = textfile.split()
stop_words = [w for w in textfile]

#Previously trained word2vec 50 dimensional vectors
fname = "/Users/semihakbayrak/ConvNet4/42bin_haber_w2v_2"
model = gensim.models.Word2Vec.load(fname)
vocab = list(model.vocab.keys())

#read the pretrained weights and save to weightDict
for category in categories:
	textfile = open(directory+'/'+category+'_weights.txt','r').read()
	textfile = textfile.split()
	weightDict[category+'_u1'] = np.array([])
	weightDict[category+'_u2'] = np.array([])
	weightDict[category+'_u3'] = np.array([])
	weightDict[category+'_u0'] = np.array([])
	weightDict[category+'_v1'] = np.array([])
	weightDict[category+'_v2'] = np.array([])
	weightDict[category+'_v3'] = np.array([])
	weightDict[category+'_v4'] = np.array([])
	weightDict[category+'_v0'] = np.array([])
	weightDict[category+'_w1'] = np.array([])
	weightDict[category+'_w2'] = np.array([])
	weightDict[category+'_w3'] = np.array([])
	weightDict[category+'_w4'] = np.array([])
	weightDict[category+'_w5'] = np.array([])
	weightDict[category+'_w0'] = np.array([])
	weightDict[category+'_p'] = np.array([])
	weightDict[category+'_p0'] = np.array([])
	for i in range(len(textfile)):
		if i<50:
			weightDict[category+'_u1'] = np.append(weightDict[category+'_u1'],float(textfile[i]))
		elif i<100:
			weightDict[category+'_u2'] = np.append(weightDict[category+'_u2'],float(textfile[i]))
		elif i<150:
			weightDict[category+'_u3'] = np.append(weightDict[category+'_u3'],float(textfile[i]))
		elif i==150:
			weightDict[category+'_u0'] = np.append(weightDict[category+'_u0'],float(textfile[i]))
		elif i<201:
			weightDict[category+'_v1'] = np.append(weightDict[category+'_v1'],float(textfile[i]))
		elif i<251:
			weightDict[category+'_v2'] = np.append(weightDict[category+'_v2'],float(textfile[i]))
		elif i<301:
			weightDict[category+'_v3'] = np.append(weightDict[category+'_v3'],float(textfile[i]))
		elif i<351:
			weightDict[category+'_v4'] = np.append(weightDict[category+'_v4'],float(textfile[i]))
		elif i==351:
			weightDict[category+'_v0'] = np.append(weightDict[category+'_v0'],float(textfile[i]))
		elif i<402:
			weightDict[category+'_w1'] = np.append(weightDict[category+'_w1'],float(textfile[i]))
		elif i<452:
			weightDict[category+'_w2'] = np.append(weightDict[category+'_w2'],float(textfile[i]))
		elif i<502:
			weightDict[category+'_w3'] = np.append(weightDict[category+'_w3'],float(textfile[i]))
		elif i<552:
			weightDict[category+'_w4'] = np.append(weightDict[category+'_w4'],float(textfile[i]))
		elif i<602:
			weightDict[category+'_w5'] = np.append(weightDict[category+'_w5'],float(textfile[i]))
		elif i==602:
			weightDict[category+'_w0'] = np.append(weightDict[category+'_w0'],float(textfile[i]))
		elif i<618:
			weightDict[category+'_p'] = np.append(weightDict[category+'_p'],float(textfile[i]))
		elif i==618:
			weightDict[category+'_p0'] = np.append(weightDict[category+'_p0'],float(textfile[i]))

def func_sigmoid(x):
	f = 1.0/(1.0+np.exp(-x))
	return f

def forward_conv3_prop(M,weight1,weight2,weight3,weight0):
	b = np.zeros((len(M)-2))
	for i in range(len(b)):
		b[i] = np.inner(M[i],weight1) + np.inner(M[i+1],weight2) + np.inner(M[i+2],weight3) + weight0[0]
	c = func_sigmoid(b)
	return c

def forward_conv4_prop(M,weight1,weight2,weight3,weight4,weight0):
	b = np.zeros((len(M)-3))
	for i in range(len(b)):
		b[i] = np.inner(M[i],weight1) + np.inner(M[i+1],weight2) + np.inner(M[i+2],weight3) + np.inner(M[i+3],weight4) + weight0[0]
	c = func_sigmoid(b)
	return c

def forward_conv5_prop(M,weight1,weight2,weight3,weight4,weight5,weight0):
	b = np.zeros((len(M)-4))
	for i in range(len(b)):
		b[i] = np.inner(M[i],weight1) + np.inner(M[i+1],weight2) + np.inner(M[i+2],weight3) + np.inner(M[i+3],weight4) + np.inner(M[i+4],weight5) + weight0[0]
	c = func_sigmoid(b)
	return c


num_of_docs = 0
num_of_correct_classified_docs = 0
num_of_topic_detected_docs = 0

for cat in categories:
	directory_cat = directory+'/test/'+cat
	textlist = []
	for filename in os.listdir(directory_cat):
		textlist.append(filename)
	for text in textlist:
		if text == '.DS_Store':
			textlist.remove('.DS_Store')
	for i in range(50):
		textfile = open(directory_cat+'/'+textlist[i],'r').read()
		textfile = textfile.decode('utf-8','ignore')
		doclist = [ line for line in textfile ]
		docstr = '' . join(doclist)
		sentences = re.split(r'[.!?]', docstr)
		documentVector = np.array([]) #document level array
		documentIndexVector = np.array([]) #to keep index values of max vals for each sentences
		documentVector3 = np.array([]) #document level array for 3 window size filter
		documentIndexVector3 = np.array([]) #to keep index values of max vals for each sentences for 3 window size filter
		documentVector4 = np.array([]) #document level array for 4 window size filter
		documentIndexVector4 = np.array([]) #to keep index values of max vals for each sentences for 4 window size filter
		documentVector5 = np.array([]) #document level array for 5 window size filter
		documentIndexVector5 = np.array([]) #to keep index values of max vals for each sentences for 5 window size filter
		docVoc = {} #to keep documentVectors for different classes
		doc3Voc = {} #to keep documentVectors for different classes
		doc4Voc = {} #to keep documentVectors for different classes
		doc5Voc = {} #to keep documentVectors for different classes
		#creating DocVoc arrays
		for category in categories:
			doc3Voc[category] = np.array([])
			doc4Voc[category] = np.array([])
			doc5Voc[category] = np.array([])
			docVoc[category] = np.array([])
		#going over all sentences in the document
		for s in range(len(sentences)):
			sentence = sentences[s]
			sentence = sentence.lower() #lower all the characters
			sentence = re.sub(r"\d+",'',sentence,flags=re.U)    #remove numbers	
			sentence = re.sub(r"\W+",'\n',sentence,flags=re.U)  #remove non alphanumerics with new line
			sentence = sentence.split() #array formed by words in the sentence
			words = [w for w in sentence if w not in stop_words] #eliminating stop words
			sentence = ' '.join(words)
			wordlist = sentence.split() #array formed by words in the sentence
			sentenceMatrix = np.empty((0,50),float) #sentence level matrix
			for word_index in range(len(wordlist)):
				if len(wordlist[word_index])>=2:
					if (wordlist[word_index]) in vocab:
						sentenceMatrix = np.append(sentenceMatrix,[model[wordlist[word_index]]],axis=0) #filling matrix with word vectors
			#forward propagation in sentence level
			if len(sentenceMatrix)>=6:
				for category in categories:
					c3_array = forward_conv3_prop(sentenceMatrix,weightDict[category+'_u1'],weightDict[category+'_u2'],weightDict[category+'_u3'],weightDict[category+'_u0'])
					c4_array = forward_conv4_prop(sentenceMatrix,weightDict[category+'_v1'],weightDict[category+'_v2'],weightDict[category+'_v3'],weightDict[category+'_v4'],weightDict[category+'_v0'])
					c5_array = forward_conv5_prop(sentenceMatrix,weightDict[category+'_w1'],weightDict[category+'_w2'],weightDict[category+'_w3'],weightDict[category+'_w4'],weightDict[category+'_w5'],weightDict[category+'_w0'])
					#max-pooling
					m3 = c3_array.max()
					m4 = c4_array.max()
					m5 = c5_array.max()
					doc3Voc[category] = np.append(doc3Voc[category],m3) #filling corresponding array with sentence representations
					doc4Voc[category] = np.append(doc4Voc[category],m4) #filling corresponding array with sentence representations
					doc5Voc[category] = np.append(doc5Voc[category],m5) #filling corresponding array with sentence representations
		posterior_probabilities = np.array([])
		for category in categories:
			docVoc[category] = np.concatenate((doc3Voc[category],doc4Voc[category]),axis=0)
			docVoc[category] = np.concatenate((docVoc[category],doc5Voc[category]),axis=0)
			if len(docVoc[category])>=21:
				#max-5-pooling in document level
				s3_index = doc3Voc[category].argsort()[-5:][::-1]
				s3_array = np.zeros((5))
				s4_index = doc4Voc[category].argsort()[-5:][::-1]
				s4_array = np.zeros((5))
				s5_index = doc5Voc[category].argsort()[-5:][::-1]
				s5_array = np.zeros((5))
				for j in range(5):
					s3_array[j] = doc3Voc[category][s3_index[j]]
					s4_array[j] = doc4Voc[category][s4_index[j]]
					s5_array[j] = doc5Voc[category][s5_index[j]]
				s_array = np.concatenate((s3_array,s4_array),axis=0)
				s_array = np.concatenate((s_array,s5_array),axis=0)
				s_index = np.concatenate((s3_index,s4_index),axis=0)
				s_index = np.concatenate((s_index,s5_index),axis=0)
				o = np.inner(s_array,weightDict[category+'_p']) + weightDict[category+'_p0'][0]
				y = func_sigmoid(o)
				posterior_probabilities = np.append(posterior_probabilities,y) #add sigmoid results of each topics' filter banks
				if category == cat:
					if y >= 0.5:
						num_of_topic_detected_docs = num_of_topic_detected_docs + 1 #if sigmoid greater than 0.5, this means topic detected
				print cat
				print category
				print y
		print '************************************'
		if len(posterior_probabilities)>0:
			num_of_docs = num_of_docs + 1
			if categories[np.argmax(posterior_probabilities)] == cat:
				num_of_correct_classified_docs = num_of_correct_classified_docs + 1 #if max posterior belongs to correct category


#accuracy rates
topic_detection_accuracy = 1.0*num_of_topic_detected_docs/num_of_docs
overall_accuracy = 1.0*num_of_correct_classified_docs/num_of_docs

print topic_detection_accuracy
print overall_accuracy
			



