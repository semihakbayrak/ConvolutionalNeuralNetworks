#inputCreator3.py
#this script takes pozitive and negative instances for one class randomly and shuffle them

import os
from shutil import copyfile
from random import shuffle
import numpy as np

class InOut:
	def __init__(self,direc,trainedCat):
		self.directory = direc
		self.trainedCat = trainedCat
	def inout_return(self):
		categorylist = []
		for filename in os.listdir(self.directory):
			categorylist.append(filename)
		for category in categorylist:
			if category == '.DS_Store':
				categorylist.remove('.DS_Store')

		inp = []
		output = []
		for category in categorylist:
			directory_category = self.directory+'/'+str(category)
			textlist = []
			count = 0
			for filename in os.listdir(directory_category):
				textlist.append(filename)
			for text in textlist:
				if text == '.DS_Store':
					textlist.remove('.DS_Store')
				else:
					if str(category) == self.trainedCat:
						inp.append([category, text])
						output.append(1.0)
						if count == 800:
							break
					else:
						inp.append([category, text])
						output.append(0.0)
						if count == 80:
							break
				count = count+1
		
		list1_shuf = []
		list2_shuf = []
		index_shuf = range(len(inp))
		shuffle(index_shuf)
		for i in index_shuf:
			list1_shuf.append(inp[i])
			list2_shuf.append(output[i])
		return list1_shuf,list2_shuf