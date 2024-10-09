### This is a demo code that demonstraate how the Regression algorithm works in predicting the final results  
# while also analyzing the weights

import numpy as np
import matplotlib as plt
import torch
from numpy import ones,vstack
from numpy.linalg import lstsq
from scipy.optimize import curve_fit
import pandas as pd
import seaborn as sns
import matplotlib.ticker as plticker

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from optigon.dataframe import *

###
## Input Data
###

input_pl_data = pd.read_csv('/Users/rqzhang/Desktop/ML/input_pl_data_wo_outlier_new_dataset.csv', sep=',', header=None).to_numpy()
input_trans_data = pd.read_csv('/Users/rqzhang/Desktop/ML/input_trans_data_wo_outlier_new_dataset.csv', sep=',', header=None).to_numpy()
input_trpl_data = pd.read_csv('/Users/rqzhang/Desktop/ML/input_trpl_data_wo_outlier_new_dataset.csv', sep=',', header=None).to_numpy()
input = np.concatenate((input_trans_data, input_pl_data, input_trpl_data), axis = -1)
print(input.shape)

test_pl_data = pd.read_csv('/Users/rqzhang/Desktop/ML/valid_pl_data_wo_outlier_new_dataset.csv', sep=',', header=None).to_numpy()
test_trans_data = pd.read_csv('/Users/rqzhang/Desktop/ML/valid_trans_data_wo_outlier_new_dataset.csv', sep=',', header=None).to_numpy()
test_trpl_data = pd.read_csv('/Users/rqzhang/Desktop/ML/valid_trpl_data_wo_outlier_new_dataset.csv', sep=',', header=None).to_numpy()
test_input_set = np.concatenate((test_trans_data, test_pl_data, test_trpl_data), axis = -1)
print(test_input_set.shape)

y = pd.read_csv('/Users/rqzhang/Desktop/ML/Output_Batch_Data/212_output_Train.csv', sep=',', header=None)
y = np.array(y)
Voc = y[:,0]
Jsc = y[:,1]
FF = y[:,2]

output_test = []
output_v = pd.read_csv('/Users/rqzhang/Desktop/ML/Output_Batch_Data/212_output_Validation.csv', sep=',', header=None)
output_test = np.array(output_v.values)
print('Test Output:')
print(output_test)
print("Finish Loading Test Output")


###
## Model
###

model = LinearRegression().fit(input, Voc) # Voc, Jsc and FF individually predict one at a time
r_sq = model.score(input, Voc)
print(f"coefficient of determination: {r_sq}")
print(f"intercept: {model.intercept_}")
print(f"coefficients: {model.coef_}")
print(model.coef_.shape)

Voc_pred = model.predict(test_input_set)
print(f"predicted response:\n{Voc_pred}")
error = (Voc_pred - output_test[:,0])/output_test[:,0]*100 # 0 for Voc, 1 for Jsc, 2 for FF
print(error)
print('Percent error is:', np.average(abs(error)), '%')

