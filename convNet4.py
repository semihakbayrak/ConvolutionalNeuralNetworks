#-*- coding: utf-8 -*- 

#convNet4.py
#training part of CNN

from inputCreator3 import InOut
import os
import re
import locale
from collections import Counter
import numpy as np
import gensim

#Preperation of inputs outputs
directory = '/Users/semihakbayrak/ConvNet4/training'
category_to_be_trained = 'spor'
objInOut = InOut(directory,category_to_be_trained)
inp_train,out_train = objInOut.inout_return()

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

def weight_init(dim):
	w = np.random.uniform(-0.01,0.01,dim)
	return w

def func_sigmoid(x):
	f = 1.0/(1.0+np.exp(-x))
	return f

def func_threshold(x):
	if x>=0:
		return 1.0
	else:
		return 0

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

def training(inp,out,numepoch,direc,stopWords,nu1=0.1,nu2=0.2):
	#sentence level weight initialization for filter with window size 3
	u1 = weight_init(50)
	u2 = weight_init(50)
	u3 = weight_init(50)
	u0 = weight_init(1)
	#sentence level weight initialization for filter with window size 4
	v1 = weight_init(50)
	v2 = weight_init(50)
	v3 = weight_init(50)
	v4 = weight_init(50)
	v0 = weight_init(1)
	#sentence level weight initialization for filter with window size 5
	w1 = weight_init(50)
	w2 = weight_init(50)
	w3 = weight_init(50)
	w4 = weight_init(50)
	w5 = weight_init(50)
	w0 = weight_init(1)
	#last layer weight initialization
	p = weight_init(15)
	p0 = weight_init(1)[0]
	for epoch in range(numepoch):
		count_valid_docs = 0
		y_store = np.array([])
		r_store = np.array([])
		#going over all shuffled documents in the training set
		for doc in range(len(inp)):
			documentVector = np.array([]) #document level array
			documentIndexVector = np.array([]) #to keep index values of max vals for each sentences
			documentVector3 = np.array([]) #document level array for 3 window size filter
			documentIndexVector3 = np.array([]) #to keep index values of max vals for each sentences for 3 window size filter
			documentVector4 = np.array([]) #document level array for 4 window size filter
			documentIndexVector4 = np.array([]) #to keep index values of max vals for each sentences for 4 window size filter
			documentVector5 = np.array([]) #document level array for 5 window size filter
			documentIndexVector5 = np.array([]) #to keep index values of max vals for each sentences for 5 window size filter
			sM_keeper = {} #to keep sentence metrices, will be used during back propagation
			cA3_keeper = {} #to keep c arrays, will be used during back propagation
			cA4_keeper = {} #to keep c arrays, will be used during back propagation
			cA5_keeper = {} #to keep c arrays, will be used during back propagation
			doc_dir = direc+'/'+str(inp[doc][0])
			doc_name = doc_dir+'/'+str(inp[doc][1])
			textfile = open(doc_name,'r').read()
			textfile = textfile.decode('utf-8','ignore')
			doclist = [ line for line in textfile ]
			docstr = '' . join(doclist)
			sentences = re.split(r'[.!?]', docstr)
			#going over all sentences in the document
			count_valid_sentences = 0
			for s in range(len(sentences)):
				sentence = sentences[s]
				sentence = sentence.lower() #lower all the characters
				sentence = re.sub(r"\d+",'',sentence,flags=re.U)    #remove numbers	
				sentence = re.sub(r"\W+",'\n',sentence,flags=re.U)  #remove non alphanumerics with new line
				sentence = sentence.split() #array formed by words in the sentence
				words = [w for w in sentence if w not in stopWords] #eliminating stop words
				sentence = ' '.join(words)
				wordlist = sentence.split() #array formed by words in the sentence
				sentenceMatrix = np.empty((0,50),float) #sentence level matrix
				for word_index in range(len(wordlist)):
					if len(wordlist[word_index])>=2:
						if (wordlist[word_index]) in vocab:
							sentenceMatrix = np.append(sentenceMatrix,[model[wordlist[word_index]]],axis=0) #filling matrix with word vectors
				#forward propagation in sentence level
				if len(sentenceMatrix)>=6:
					c3_array = forward_conv3_prop(sentenceMatrix,u1,u2,u3,u0)
					c4_array = forward_conv4_prop(sentenceMatrix,v1,v2,v3,v4,v0)
					c5_array = forward_conv5_prop(sentenceMatrix,w1,w2,w3,w4,w5,w0)
					#max-pooling
					m3 = c3_array.max()
					m3_index = c3_array.argmax()
					m4 = c4_array.max()
					m4_index = c4_array.argmax()
					m5 = c5_array.max()
					m5_index = c5_array.argmax()
					documentVector3 = np.append(documentVector3,m3) #filling array with sentence representations
					documentIndexVector3 = np.append(documentIndexVector3,m3_index) #filling array with indices of max vals
					documentVector4 = np.append(documentVector4,m4) #filling array with sentence representations
					documentIndexVector4 = np.append(documentIndexVector4,m4_index) #filling array with indices of max vals
					documentVector5 = np.append(documentVector5,m5) #filling array with sentence representations
					documentIndexVector5 = np.append(documentIndexVector5,m5_index) #filling array with indices of max vals
					sM_keeper[count_valid_sentences] = sentenceMatrix #keep valid sentence matrix for back propagation
					cA3_keeper[count_valid_sentences] = c3_array[m3_index] #keep max c vals for back propagation
					cA4_keeper[count_valid_sentences] = c4_array[m4_index] #keep max c vals for back propagation
					cA5_keeper[count_valid_sentences] = c5_array[m5_index] #keep max c vals for back propagation
					count_valid_sentences = count_valid_sentences + 1
			documentVector = np.concatenate((documentVector3,documentVector4),axis=0)
			documentVector = np.concatenate((documentVector,documentVector5),axis=0)
			if len(documentVector)>=21:
				#max-5-pooling in document level
				s3_index = documentVector3.argsort()[-5:][::-1]
				s3_array = np.zeros((5))
				s4_index = documentVector4.argsort()[-5:][::-1]
				s4_array = np.zeros((5))
				s5_index = documentVector5.argsort()[-5:][::-1]
				s5_array = np.zeros((5))
				for i in range(5):
					s3_array[i] = documentVector3[s3_index[i]]
					s4_array[i] = documentVector4[s4_index[i]]
					s5_array[i] = documentVector5[s5_index[i]]
				s_array = np.concatenate((s3_array,s4_array),axis=0)
				s_array = np.concatenate((s_array,s5_array),axis=0)
				s_index = np.concatenate((s3_index,s4_index),axis=0)
				s_index = np.concatenate((s_index,s5_index),axis=0)
				o = np.inner(s_array,p) + p0
				y = func_sigmoid(o)
				#backpropagation
				r = out[doc]
				#delta_p
				delta_p = nu1*(r-y)*s_array
				delta_p0 = nu1*(r-y)
				#delta_u, delta_v, delta_w
				delta_u1 = np.zeros((50))
				delta_u2 = np.zeros((50))
				delta_u3 = np.zeros((50))
				delta_u0 = 0
				delta_v1 = np.zeros((50))
				delta_v2 = np.zeros((50))
				delta_v3 = np.zeros((50))
				delta_v4 = np.zeros((50))
				delta_v0 = 0
				delta_w1 = np.zeros((50))
				delta_w2 = np.zeros((50))
				delta_w3 = np.zeros((50))
				delta_w4 = np.zeros((50))
				delta_w5 = np.zeros((50))
				delta_w0 = 0
				for h in range(15):
					j = s_index[h]
					sM = sM_keeper[j]
					if h<5:
						cA = cA3_keeper[j]
						i = documentIndexVector3[j]
						delta_u1 = delta_u1 + nu2*(r-y)*p[h]*cA*(1-cA)*sM[i]
						delta_u2 = delta_u2 + nu2*(r-y)*p[h]*cA*(1-cA)*sM[i+1]
						delta_u3 = delta_u3 + nu2*(r-y)*p[h]*cA*(1-cA)*sM[i+2]
						delta_u0 = delta_u0 + nu2*(r-y)*p[h]*cA*(1-cA)
					elif h<10:
						cA = cA4_keeper[j]
						i = documentIndexVector4[j]
						delta_v1 = delta_v1 + nu2*(r-y)*p[h]*cA*(1-cA)*sM[i]
						delta_v2 = delta_v2 + nu2*(r-y)*p[h]*cA*(1-cA)*sM[i+1]
						delta_v3 = delta_v3 + nu2*(r-y)*p[h]*cA*(1-cA)*sM[i+2]
						delta_v4 = delta_v4 + nu2*(r-y)*p[h]*cA*(1-cA)*sM[i+3]
						delta_v0 = delta_v0 + nu2*(r-y)*p[h]*cA*(1-cA)
					else:
						cA = cA5_keeper[j]
						i = documentIndexVector5[j]
						delta_w1 = delta_w1 + nu2*(r-y)*p[h]*cA*(1-cA)*sM[i]
						delta_w2 = delta_w2 + nu2*(r-y)*p[h]*cA*(1-cA)*sM[i+1]
						delta_w3 = delta_w3 + nu2*(r-y)*p[h]*cA*(1-cA)*sM[i+2]
						delta_w4 = delta_w4 + nu2*(r-y)*p[h]*cA*(1-cA)*sM[i+3]
						delta_w5 = delta_w5 + nu2*(r-y)*p[h]*cA*(1-cA)*sM[i+4]
						delta_w0 = delta_w0 + nu2*(r-y)*p[h]*cA*(1-cA)
				#updating weights
				u1 = u1 + delta_u1
				u2 = u2 + delta_u2
				u3 = u3 + delta_u3
				u0 = u0 + delta_u0
				v1 = v1 + delta_v1
				v2 = v2 + delta_v2
				v3 = v3 + delta_v3
				v4 = v4 + delta_v4
				v0 = v0 + delta_v0
				w1 = w1 + delta_w1
				w2 = w2 + delta_w2
				w3 = w3 + delta_w3
				w4 = w4 + delta_w4
				w5 = w5 + delta_w5
				w0 = w0 + delta_w0
				p = p + delta_p
				p0 = p0 + delta_p0
				#store y and r values
				y_store = np.append(y_store,y)
				r_store = np.append(r_store,r)
				count_valid_docs = count_valid_docs + 1
				print r
				print y
				print doc
			if count_valid_docs==50:
				cross_entropy = -1*np.inner(r_store,np.log(y_store)) #Cross-Entropy
				count_valid_docs = 0
				y_store = np.array([])
				r_store = np.array([])
				print "CROSS-ENTROPY"
				print cross_entropy
	#save the found weights to txt file
	with open(category_to_be_trained+'_weights.txt',"w") as cat_f:
		cat_f.write("\n".join(" ".join(map(str, x)) for x in (u1,u2,u3,u0,v1,v2,v3,v4,v0,w1,w2,w3,w4,w5,w0,p,np.array([p0]))))



training(inp_train,out_train,1,directory,stop_words)
