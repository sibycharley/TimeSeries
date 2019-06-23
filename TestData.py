import sys
sys.path.append("C:\\Anaconda3\\envs\\tensorflow\\python35.zip")
sys.path.append("C:\\Anaconda3\\envs\\tensorflow\\DLLs")
sys.path.append("C:\\Anaconda3\\envs\\tensorflow\\lib") 
sys.path.append("C:\\Anaconda3\\envs\\tensorflow") 
sys.path.append("C:\\Anaconda3\\envs\\tensorflow\\lib\\site-packages")
sys.path.append("C:\\Anaconda3\\envs\\tensorflow\\lib\\site-packages\\setuptools-27.2.0-py3.5.egg")
print(sys.path)

import csv
import numpy as np
import pandas as pd
import scipy
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf

from random import random
from random import randint
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import accuracy_score

from keras.utils import np_utils, to_categorical
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Masking, LSTM, Dense, Dropout
from keras.models import load_model
from keras.layers import Bidirectional, Activation
from keras import regularizers, optimizers
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape

class DTest():

	#Initialization
	def __init__(self, test_data, test_label):
		self.tdata = test_data
		self.tlabel = test_label


	#Cleaning and Stratification
	def Test_TimeSeries(self):

		#Reading the Trainig and Validation files
		dF1 = pd.read_csv(self.tdata)
		dF2 = pd.read_csv(self.tlabel)

		XTest = dF1.values
		XTest = XTest.reshape((XTest.shape[0], XTest.shape[1], 1))	
		YTest = dF2.values
		YTest = to_categorical(YTest, num_classes=4)

		#Constants
		step = 10
		vecLen = 1
		bSize = 128

		#Loading the trained model
		lMod = load_model('TimeSeries_LSTM2.h5')

		#Predicting the target for Test Data
		val = lMod.predict(XTest, batch_size=bSize, verbose=2)		
		res = val.argmax(axis=-1)

		#Metrics
		print (accuracy_score(dF2.values, res))
		print (roc_auc_score(YTest, val))
		print (f1_score(dF2.values, res, average='macro'))


if __name__ == "__main__":

	#Define the file paths and directories
	tData = 'Test_data.csv'
	tLab = 'Test_label.csv'

	#Call the Cleaning constructor
	DTs = DTest(tData, tLab)

	#Call the Feature Extraction Execution
	DTs.Test_TimeSeries()	

	print ("Hello")					