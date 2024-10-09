import numpy as np
import matplotlib as plt
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import random
import pdb
import os

from optigon.dataframe import *
from optigon.mockdata.generation import generate_dataframe


###
## Demo List
###

end = [32, 36, 28, 36, 28, 36, 32, 16]
outlier = [(1,2), (2,11), (2,25), (2,26), (2,27), (2,28), (4,1), (4,2), (4,3), (4,4), (4,17), (4,18), (4,19), (4,20), (4,29), (8,1), (8,2), (8,3), (8,4), (8,5), (8,9), (8,10), (8,11), (8,12)]
outlier_num = [2, 43, 57, 58, 59, 60, 97, 98, 99, 100, 113, 114, 115, 116, 125, 229, 230, 231, 232, 233, 237, 238, 239, 240]
random_list = [4, 16, 34, 68, 83, 143, 177, 216, 217, 224]
test_list = [(1, 4), (1, 16), (2, 2), (2, 36),(3, 15), (5, 11), (6, 17), (7, 20), (7, 21), (7, 28)]

## Could Generate Random List
# random_list = []
# count = 0
# test_list = []
# while count < 10:
#     r = random.randint(1,244)
#     if r not in random_list and r not in outlier_num:
#         random_list.append(r)
#         count+=1
#         total = 0
#         for i in range(len(end)):
#             total += end[i]
#             if r <= total:
#                 test_list.append((i+1, r-total+end[i]))
#                 break

# random_list.sort()
# test_list.sort()

print(random_list)
print(test_list)


###
## Load Training Input_file in type of Trans, PL, TrPL
###

pl_numpy_input = []
trpl_numpy_input = []
trans_numpy_input = []

for i in range (1,9):
    for j in range (1,end[i-1]+1):
        if (i,j) not in test_list and (i,j) not in outlier:
            df = DataFrame.from_file('/Users/rqzhang/Desktop/ML/Input_Batch_Data_234/gridedge_' + str(i) + '_' + str(j) + '.df')
            pl_numpy_data = extract_numpy(df, "pl", normalize=True)
            trans_numpy_data = extract_numpy(df, "trans")
            trpl_numpy_data = extract_numpy(df,"trpl")
          
# Trans
            trans_numpy_y = trans_numpy_data[1,1500:2500] 
            input_trans_data = []
            for v in range(1000):
                if v % 4 == 1:
                    input_trans_data.append(trans_numpy_y[v])

# PL
            pl_numpy_y = pl_numpy_data[1,1750:2750]
            input_pl_data = []
            for k in range(1000):
                if k % 4 == 1:
                    input_pl_data.append(pl_numpy_y[k])

# TrPL
            trpl_numpy_y = trpl_numpy_data[1,0:500]
            input_trpl_data = []
            for w in range(500):
                if w % 2 == 1:
                    input_trpl_data.append(trpl_numpy_y[w])
          
            trans_numpy_input.append(np.array(input_trans_data))
            pl_numpy_input.append(np.array(input_pl_data))
            trpl_numpy_input.append(np.array(input_trpl_data))

pl_numpy_input = np.array(pl_numpy_input)
trans_numpy_input=np.array(trans_numpy_input)
trpl_numpy_input=np.array(trpl_numpy_input)

print('trans data shape is:', np.shape(trans_numpy_input))
print('pl data shape is:', np.shape(pl_numpy_input))
print('trpl data shape is:', np.shape(trpl_numpy_input))

input_pl_data = torch.from_numpy(pl_numpy_input).float()
input_trans_data = torch.from_numpy(trans_numpy_input).float()
input_trpl_data = torch.from_numpy(trpl_numpy_input).float()
print("Finish Loading training Input")


###
## Load Training output Voc Jsc FF
###

output_truth = []
output = pd.read_csv('/Users/rqzhang/Desktop/ML/Output_Batch_Data/244_output_full.csv', sep=',', header=None)
print(output.shape)
np_output = np.array(output.values)
for i in range(len(np_output)):
    if i not in random_list and i not in outlier_num:
        output_truth.append(np_output[i,:])

output_truth =np.array(output_truth)
print(np.shape(output_truth))

output_truth = torch.from_numpy(output_truth).float()
print("Finish Loading Training Ground Truth Output")


###
## Load Testing input
###

test_trans_data = []
test_pl_data = []
test_trpl_data = []

for i in range (1,9):
    for j in range (1,end[i-1]+1):
        if (i,j) in test_list and (i,j) not in outlier:
            df = DataFrame.from_file('/Users/rqzhang/Desktop/ML/Input_Batch_Data_234/gridedge_' + str(i) + '_' + str(j) + '.df')
            test_pl_data_in = extract_numpy(df, "pl", normalize=True)
            test_trans_data_in = extract_numpy(df, "trans")
            test_trpl_data_in = extract_numpy(df,"trpl")

# Trans
            trans_test_numpy_y = test_trans_data_in[1,1500:2500] # 2000 points, wavelength from 650-850nm
            test_input_trans_data = []
            for v in range(1000):
                if v % 4 == 1:
                    test_input_trans_data.append(trans_test_numpy_y[v])

        # PL
            pl_test_numpy_y = test_pl_data_in[1,1750:2750] # 2000 points, wavelength from 700-900nm
            test_input_pl_data = []
            for k in range(1000):
                if k % 4 == 1:
                    test_input_pl_data.append(pl_test_numpy_y[k])

        # TrPL
            trpl_test_numpy_y = test_trpl_data_in[1,0:500] # first 500 points
            test_input_trpl_data = []
            for w in range(500):
                if w % 2 == 1:
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
print("Finish Loading Test Input")


###
## Load test output
###
output_test = []


for j in range(len(np_output)):
    if j in random_list:
        output_test.append(np_output[j,:])

output_test = np.array(output_test)
print(np.shape(output_test))
print(output_test)
output_test = torch.from_numpy(output_test).float()
print("Finish Loading Test Output")


###
## Model
###

input_dim_trans = 250
input_dim_pl = 250
input_dim_trpl = 250
output_dim = 3

## Neural Network Class
class NeuralNetwork(nn.Module):

    # Design Layer
    def __init__(self, input_dim_trans, input_dim_pl, input_dim_trpl, output_dim):
        super(NeuralNetwork,self).__init__()

        # Trans
        self.layer_1 = nn.Linear(input_dim_trans, 100)
        self.layer_2 = nn.Linear(100, 50)
        self.layer_3 = nn.Linear(50,output_dim)

        # PL
        self.layer_4 = nn.Linear(input_dim_pl, 100)
        self.layer_5 = nn.Linear(100, 50)
        self.layer_6 = nn.Linear(50,output_dim)

        # TrPL
        self.layer_7 = nn.Linear(input_dim_trpl, 100)
        self.layer_8 = nn.Linear(100, 50)
        self.layer_9 = nn.Linear(50,output_dim)

        # In all
        self.layer_10 = nn.Linear(9,output_dim)
        
    def forward(self, input_trans_data, input_pl_data, input_trpl_data):
        # Trans
        trans = input_trans_data
        trans = torch.nn.functional.relu(self.layer_1(trans))
        trans = torch.nn.functional.relu(self.layer_2(trans))
        trans = torch.nn.functional.relu(self.layer_3(trans))

        # PL
        PL = input_pl_data
        PL = torch.nn.functional.relu(self.layer_4(PL))
        PL = torch.nn.functional.relu(self.layer_5(PL))
        PL = torch.nn.functional.relu(self.layer_6(PL))

        # TrPL
        TrPL = input_trpl_data
        TrPL = torch.nn.functional.relu(self.layer_7(TrPL))
        TrPL = torch.nn.functional.relu(self.layer_8(TrPL))
        TrPL = torch.nn.functional.relu(self.layer_9(TrPL))

        cat = torch.cat((trans, PL, TrPL), axis = -1)
        output = self.layer_10(cat)

        return output

model = NeuralNetwork(input_dim_trans, input_dim_pl, input_dim_trpl, output_dim).float()

## Loss Function
learning_rate = 0.0001
# loss_fn = nn.MSELoss() # l2-norm
loss_fn = nn.L1Loss() # l1-norm
# optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

## Training and testing Loops
loss_values = []
test_loss_value = []

for epoch in range(15000):

    # Test
    if epoch % 1000 == 0:
        print(epoch)

    final_test = model(test_trans_data, test_pl_data, test_trpl_data)
    test_loss = loss_fn(final_test, output_test.squeeze(0))
    if epoch % 1000 == 0:
        print("Test loss:", test_loss.item())
    test_loss_value.append(test_loss.item())


    

    #Training
    if epoch % 1000 == 0:
        print(epoch)
    optimizer.zero_grad()

    pred = model(input_trans_data, input_pl_data, input_trpl_data)
    loss = loss_fn(pred, output_truth.squeeze(0))
    if epoch % 1000 == 0:
        print("Training loss:", loss.item())
    loss_values.append(loss.item())

    loss.backward()
    optimizer.step()

print("Training Complete")
print("Final Training Loss:", loss.item())
print('Test is', final_test)

final_test.numpy
error = abs(final_test-output_test)/output_test*100
print(error)
error = error.detach().numpy()
print(np.average(error[:,0]), np.average(error[:,1]), np.average(error[:,2]))

##track the loss of the model over time
step = np.linspace(1,15000,15000)
fig, ax = plt.subplots(figsize = (8,5))
plt.plot(step, np.array(loss_values))
plt.title("Step-wise Training Loss")
plt.xlabel("epochs")
plt.ylabel("Training Loss")
plt.show()

plt.plot(step, np.array(test_loss_value))
plt.title("Step-wise Prediction Loss")
plt.xlabel("epochs")
plt.ylabel("Prediction Loss")
plt.show()





