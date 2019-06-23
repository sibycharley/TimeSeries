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
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.metrics import precision_recall_fscore_support

from keras.utils import np_utils, to_categorical
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Masking, LSTM, Dense, Dropout
from keras.models import load_model
from keras.layers import Bidirectional, Activation
from keras import regularizers, optimizers
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape

class DTrain():

	#Initialization
	def __init__(self, inp_data, inp_label, valid_data, valid_label):
		self.idata = inp_data
		self.ilabel = inp_label

		self.vdata = valid_data
		self.vlabel = valid_label

	#Cleaning and Stratification
	def Train_TimeSeries(self):

		#Reading the Trainig and Validation files
		dF1 = pd.read_csv(self.idata)
		dF2 = pd.read_csv(self.ilabel)

		dF3 = pd.read_csv(self.vdata)
		dF4 = pd.read_csv(self.vlabel)

		XTrain = dF1.values
		XTrain = XTrain.reshape((XTrain.shape[0], XTrain.shape[1], 1))	
		YTrain = dF2.values
		YTrain = to_categorical(YTrain, num_classes=4)

		XValid = dF3.values
		XValid = XValid.reshape((XValid.shape[0], XValid.shape[1], 1))	
		YValid = dF4.values		
		YValid = to_categorical(YValid, num_classes=4)

		#Constants
		step = 10
		vecLen = 1
		bSize = 128

		#Defining the Network

		#Input Layer
		Inp = Input(shape=(step, vecLen))

		#LSTM
		Xlstm = LSTM(8)(Inp)
		XDrp = Dropout(0.2)(Xlstm)

		#Dense
		Xout = Dense(4, activation='softmax')(XDrp)

		lMod = Model(inputs=Inp, outputs=Xout)
		lMod.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0001), metrics=['accuracy'])				

		print(lMod.summary())		

		#Fitting the LSTM with the sequence data
		lMod.fit(XTrain, YTrain, validation_data=(XValid, YValid), epochs=500, batch_size=bSize, verbose=2)

		#Saving the trained model
		lMod.save('TimeSeries_LSTM.h5')		

if __name__ == "__main__":

	#Define the file paths and directories
	iData = 'Train_data.csv'
	iLab = 'Train_label.csv'

	vData = 'Valid_data.csv'
	vLab = 'Valid_label.csv'

	#Call the Cleaning constructor
	DTr = DTrain(iData, iLab, vData, vLab)

	#Call the Feature Extraction Execution
	DTr.Train_TimeSeries()	

	print ("Hello")					