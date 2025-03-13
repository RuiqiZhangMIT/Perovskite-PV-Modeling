import numpy as np
import matplotlib as plt
import torch
from numpy import ones,vstack
from numpy.linalg import lstsq
from scipy.optimize import curve_fit
import pandas as pd
import seaborn as sns
import torch.nn as nn
from torch import optim
import pdb
import os
import random

from optigon.dataframe import *
from optigon.mockdata.generation import generate_dataframe
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
from sklearn import utils
from sklearn import svm
from sklearn import preprocessing
from sklearn import utils
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


###
## Demo List
###

end = [32, 36, 28, 36, 32, 36, 32, 16, 36, 36, 32, 16]
######  1   2   3   4   5   6   7   8   9  10  11  12
random_list = [10, 30, 39, 40, 49, 66, 76, 83, 136, 155, 176, 213, 222, 261, 264, 265, 271, 294, 308, 310]
test_list = [(1, 10), (1, 30), (2, 7), (2, 8), (2, 17), (2, 34), (3, 8), (3, 15), (5, 4), (5, 23), (6, 12), (7, 13), (7, 22), (9, 13), (9, 16), (9, 17), (9, 23), (10, 10), (10, 24), (10, 26)]
print(random_list)
print(test_list)


###
## Load Training Input
###

pl_numpy_input = []
trpl_numpy_input = []
trans_numpy_input = []

for i in range (1,13):
    for j in range (1,end[i-1]+1):
        if (i,j) not in test_list:
            df = DataFrame.from_file('/Users/rqzhang/Desktop/ML/Input_Batch_Data_Degrade_All/gridedge_' + str(i) + '_' + str(j) + '.df')
            trans_numpy_data = df.get_data_by_type("trans")[0]
            pl_numpy_data = df.get_data_by_type("srpl")[0]
            trpl_numpy_data = df.get_data_by_type("trpl")[0]
          
# Trans
            trans_numpy_y = trans_numpy_data[1,830:1580]
            input_trans_data = []
            for v in range(750):
                if v % 3 == 1:
                    input_trans_data.append(trans_numpy_y[v])

# PL
            pl_numpy_data=pl_numpy_data.astype(np.double) # Normalize PL Data
            pl_numpy_data[1] /= pl_numpy_data[1].max()

            pl_numpy_y = pl_numpy_data[1,1750:2750]
            input_pl_data = []
            for k in range(1000):
                if k % 4 == 1:
                    input_pl_data.append(pl_numpy_y[k])

# TrPL
            trpl_numpy_y = trpl_numpy_data[1,0:250]
            input_trpl_data = []
            for w in range(250):
                if w % 1 == 0:
                    input_trpl_data.append(trpl_numpy_y[w])
          
            trans_numpy_input.append(np.array(input_trans_data))
            pl_numpy_input.append(np.array(input_pl_data))
            trpl_numpy_input.append(np.array(input_trpl_data))

pl_numpy_input = np.array(pl_numpy_input)
trans_numpy_input=np.array(trans_numpy_input)
trpl_numpy_input=np.array(trpl_numpy_input)

input_pl_data = torch.from_numpy(pl_numpy_input).float()
input_trans_data = torch.from_numpy(trans_numpy_input).float()
input_trpl_data = torch.from_numpy(trpl_numpy_input).float()

input = np.concatenate((input_trans_data, input_pl_data, input_trpl_data), axis = -1)
print(input.shape)
print("Finish Loading Training Input")


###
## Load Training output Voc Jsc FF
###

output_truth = []
output = pd.read_csv('/Users/rqzhang/Desktop/ML/Output_Batch_Data/368_output_full.csv', sep=',', header=None)
np_output = np.array(output.values)
for i in range(len(np_output)):
    if i+1 not in random_list:
        output_truth.append(np_output[i,:])

output_truth =np.array(output_truth)
print(np.shape(output_truth))

for j in range(len(output_truth)):
    if output_truth[j,0] <= 700:
        output_truth[j,0] = 0
    elif output_truth[j,0] > 700 and output_truth[j,0] <= 900:
        output_truth[j,0] = 1
    elif output_truth[j,0] >= 900:
        output_truth[j,0] = 2

for k in range(len(output_truth)):
    if output_truth[k,1] <= 17:
        output_truth[k,1] = 0
    elif output_truth[k,1] > 17 and output_truth[k,1] <= 20:
        output_truth[k,1] = 1
    elif output_truth[k,1] > 20:
        output_truth[k,1] = 2

for w in range(len(output_truth)):
    if output_truth[w,2] <= 45:
        output_truth[w,2] = 0
    elif output_truth[w,2] > 45 and output_truth[w,2] <= 60:
        output_truth[w,2] = 1
    elif output_truth[w,2] > 60:
        output_truth[w,2] = 2


print(output_truth)
output_truth = torch.from_numpy(output_truth).type(torch.LongTensor)#.float()
print("Finish Loading Training Ground Truth Output")


##
## Load Testing input (Trans, PL, TrPL)
###

test_trans_data = []
test_pl_data = []
test_trpl_data = []

for i in range (1,13):
    for j in range (1,end[i-1]+1):
        if (i,j) in test_list:
            df = DataFrame.from_file('/Users/rqzhang/Desktop/ML/Input_Batch_Data_Degrade_All/gridedge_' + str(i) + '_' + str(j) + '.df')

            test_trans_data_in = df.get_data_by_type("trans")[0]
            test_pl_data_in = df.get_data_by_type("srpl")[0]
            test_trpl_data_in = df.get_data_by_type("trpl")[0]

# Trans
            trans_test_numpy_y = test_trans_data_in[1,830:1580] # 2000 points, wavelength from 638-881nm
            test_input_trans_data = []
            for v in range(750):
                if v % 3 == 1:
                    test_input_trans_data.append(trans_test_numpy_y[v])

        # PL
            test_pl_data_in=test_pl_data_in.astype(np.double) # Normalize PL Data
            test_pl_data_in[1] /= test_pl_data_in[1].max()

            pl_test_numpy_y = test_pl_data_in[1,1750:2750] # 1000 points, wavelength from 703-886nm
            test_input_pl_data = []
            for k in range(1000):
                if k % 4 == 1:
                    test_input_pl_data.append(pl_test_numpy_y[k])

        # TrPL
            trpl_test_numpy_y = test_trpl_data_in[1,0:250] # first 250 points
            test_input_trpl_data = []
            for w in range(250):
                if w % 1 == 0:
                    test_input_trpl_data.append(trpl_test_numpy_y[w])
                
            test_trans_data.append(np.array(test_input_trans_data))
            test_pl_data.append(np.array(test_input_pl_data))
            test_trpl_data.append(np.array(test_input_trpl_data))

test_trans_data = np.array(test_trans_data)
test_pl_data=np.array(test_pl_data)
test_trpl_data=np.array(test_trpl_data)

test_trans_data = torch.from_numpy(test_trans_data).float()
test_pl_data = torch.from_numpy(test_pl_data).float()
test_trpl_data = torch.from_numpy(test_trpl_data).float()

test_input_set = np.concatenate((test_trans_data, test_pl_data, test_trpl_data), axis = -1)
print(test_input_set.shape)

print("Finish Loading Test Input")


###
## Load test output
###

output_test = []
for j in range(len(np_output)):
    if j+1 in random_list:
        output_test.append(np_output[j,:])

output_test = np.array(output_test)
print(output_test.shape)

for j in range(len(output_test)):
    if output_test[j,0] <= 700:
        output_test[j,0] = 0
    elif output_test[j,0] > 700 and output_test[j,0] <= 900:
        output_test[j,0] = 1
    elif output_test[j,0] >= 900:
        output_test[j,0] = 2

for k in range(len(output_test)):
    if output_test[k,1] <= 17:
        output_test[k,1] = 0
    elif output_test[k,1] > 17 and output_test[k,1] <= 20:
        output_test[k,1] = 1
    elif output_test[k,1] > 20:
        output_test[k,1] = 2

for w in range(len(output_test)):
    if output_test[w,2] <= 45:
        output_test[w,2] = 0
    elif output_test[w,2] > 45 and output_test[w,2] <= 60:
        output_test[w,2] = 1
    elif output_test[w,2] > 60:
        output_test[w,2] = 2

print(output_test)
output_test = torch.from_numpy(output_test).type(torch.LongTensor)#.float()
print("Finish Loading Test Output")


###
## SVM
###

X = input
Y0 = output_truth[:,0]
Y1 = output_truth[:,1]
Y2 = output_truth[:,2]
lab = preprocessing.LabelEncoder()

# clf = svm.NuSVC(gamma="auto")
clf0 = SVC(kernel="linear", gamma=0.5, C=1.0)
clf0.fit(X, Y0)
y_pred_0 = clf0.predict(test_input_set)

clf1 = SVC(kernel="linear", gamma=0.5, C=1.0)
clf1.fit(X, Y1)
y_pred_1 = clf1.predict(test_input_set)

clf2 = SVC(kernel="linear", gamma=0.5, C=1.0)
clf2.fit(X, Y2)
y_pred_2 = clf2.predict(test_input_set)

print('Voc Accuracy is: ', accuracy_score(output_test[:,0],y_pred_0))
print(output_test[:,0])
print(y_pred_0)

print('Jsc Accuracy is: ', accuracy_score(output_test[:,1],y_pred_1))
print(output_test[:,1])
print(y_pred_1)

print('FF Accuracy is: ', accuracy_score(output_test[:,2],y_pred_2))
print(output_test[:,2])
print(y_pred_2)
