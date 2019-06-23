import sys
import csv
import numpy as np
import pandas as pd
import scipy
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

class DataClean():

	#Initialization
	def __init__(self, inp):
		self.data = inp

	#Cleaning and Stratification
	def CleanStrat(self):

		#Reading the input file
		dF = pd.read_csv(self.data)

		#Constants
		step = 10
		nrow = int(dF.values.shape[0]/step)

		#Arranging the input data
		xTemp = dF.values[:,2].reshape((nrow, step))
		yTemp = dF.values[::step,3]

		#Startified splitting

		#Temp-Valid Splitting
		XTe, XVa, YTe, YVa = train_test_split(xTemp, yTemp, stratify=yTemp, test_size=0.20)
		print (XVa.shape)
		print (YVa.shape)

		#Train-Test Splitting
		XTr, XTs, YTr, YTs = train_test_split(XTe, YTe, stratify=YTe, test_size=0.20)
		print (XTr.shape)
		print (YTr.shape)

		print (XTs.shape)
		print (YTs.shape)		

		#Saving the corresponding split data into csv
		pd.DataFrame(XVa).to_csv('Valid_data.csv', index=False, header=False)
		pd.DataFrame(YVa).to_csv('Valid_label.csv', index=False, header=False)

		pd.DataFrame(XTr).to_csv('Train_data.csv', index=False, header=False)
		pd.DataFrame(YTr).to_csv('Train_label.csv', index=False, header=False)

		pd.DataFrame(XTs).to_csv('Test_data.csv', index=False, header=False)
		pd.DataFrame(YTs).to_csv('Test_label.csv', index=False, header=False)						
 

if __name__ == "__main__":

	#Define the file paths and directories
	file = 'traffic.csv'

	#Call the Cleaning constructor
	DC = DataClean(file)

	#Call the Feature Extraction Execution
	DC.CleanStrat()	

	print ("Hello")			