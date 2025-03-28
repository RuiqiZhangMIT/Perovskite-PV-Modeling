import numpy as np
import matplotlib as plt
import torch
import torch.nn as nn
from torch import optim
import pdb
import os
import pandas as pd
import random

from optigon.dataframe import *
from torch.utils.data import Dataset, DataLoader


# Generate Random Number
end = [32, 36, 28, 36, 32, 36, 32, 16, 36, 36, 32, 16]
######  1   2   3   4   5   6   7   8   9  10  11  12

random_list = []
count = 0
test_list = []
while count < 20:
    r = random.randint(1,368)
    if r not in random_list:
        random_list.append(r)
        count+=1
        total = 0
        for i in range(len(end)):
            total += end[i]
            if r <= total:
                test_list.append((i+1, r-total+end[i]))
                break

random_list.sort()
test_list.sort()

test_input = test_list

print(random_list)
print(test_list)


###
## Load Training input %T, SrPL, TrPL
###

pl_numpy_input = []
trpl_numpy_input = []
trans_numpy_input = []

for i in range (1,13):
    for j in range (1,end[i-1]+1):
        if (i,j) not in test_input:
            df = DataFrame.from_file('/Users/rqzhang/Desktop/ML/Input_Batch_Data_Degrade_All/gridedge_' + str(i) + '_' + str(j) + '.df')
            trans_numpy_data = df.get_data_by_type("trans")[0]
            pl_numpy_data = df.get_data_by_type("srpl")[0]
            trpl_numpy_data = df.get_data_by_type("trpl")[0]
          
# Trans
            trans_numpy_y = trans_numpy_data[1,830:1580] # 750 points, select wavelength range, could vary
            input_trans_data = []
            for v in range(750):
                if v % 3 == 1:
                    input_trans_data.append(trans_numpy_y[v])

# PL
            pl_numpy_data=pl_numpy_data.astype(np.double) # Normalize PL Data
            pl_numpy_data[1] /= pl_numpy_data[1].max()

            pl_numpy_y = pl_numpy_data[1,1750:2750] # 1000 points, select wavelength range, could vary
            input_pl_data = []
            for k in range(1000):
                if k % 4 == 1:
                    input_pl_data.append(pl_numpy_y[k])

# TrPL
            trpl_numpy_y = trpl_numpy_data[1,0:250] # first 250 points, could vary
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


print(np.shape(trans_numpy_input))
print(np.shape(trpl_numpy_input))
print(np.shape(pl_numpy_input))

input_pl_data = torch.from_numpy(pl_numpy_input).float()
input_trans_data = torch.from_numpy(trans_numpy_input).float()
input_trpl_data = torch.from_numpy(trpl_numpy_input).float()
print("Finish Loading training Input")


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

output_truth = torch.from_numpy(output_truth).float()
print("Finish Loading Training Ground Truth Output")


##
## Load Testing input (Trans, PL, TrPL)
###

test_trans_data = []
test_pl_data = []
test_trpl_data = []

for i in range (1,13):
    for j in range (1,end[i-1]+1):
        if (i,j) in test_input:
            df = DataFrame.from_file('/Users/rqzhang/Desktop/ML/Input_Batch_Data_Degrade_All/gridedge_' + str(i) + '_' + str(j) + '.df')

            test_trans_data_in = df.get_data_by_type("trans")[0]
            test_pl_data_in = df.get_data_by_type("srpl")[0]
            test_trpl_data_in = df.get_data_by_type("trpl")[0]

# Trans
            trans_test_numpy_y = test_trans_data_in[1,830:1580] 
            test_input_trans_data = []
            for v in range(750):
                if v % 3 == 1:
                    test_input_trans_data.append(trans_test_numpy_y[v])

        # PL
            test_pl_data_in=test_pl_data_in.astype(np.double) 
            test_pl_data_in[1] /= test_pl_data_in[1].max()

            pl_test_numpy_y = test_pl_data_in[1,1750:2750] 
            test_input_pl_data = []
            for k in range(1000):
                if k % 4 == 1:
                    test_input_pl_data.append(pl_test_numpy_y[k])

        # TrPL
            trpl_test_numpy_y = test_trpl_data_in[1,0:250] 
            test_input_trpl_data = []
            for w in range(250):
                if w % 1 == 0:
                    test_input_trpl_data.append(trpl_test_numpy_y[w])
                
            test_trans_data.append(np.array(test_input_trans_data))
            test_pl_data.append(np.array(test_input_pl_data))
            test_trpl_data.append(np.array(test_input_trpl_data))

testing_trans_data = np.array(test_trans_data)
testing_pl_data=np.array(test_pl_data)
testing_trpl_data=np.array(test_trpl_data)

test_trans_data = torch.from_numpy(testing_trans_data).float()
test_pl_data = torch.from_numpy(testing_pl_data).float()
test_trpl_data = torch.from_numpy(testing_trpl_data).float()
print("Finish Loading Testing Input")


###
## Load Testing output
###

output_test = []
for j in range(len(np_output)):
    if j+1 in random_list:
        output_test.append(np_output[j,:])

output_test = np.array(output_test)
print(np.shape(output_test))
print(output_test)
output_test = torch.from_numpy(output_test).float()
print("Finish Loading Testing Output")
# np.savetxt(r'/Users/rqzhang/Desktop/Groud_Output.csv', output_test, delimiter=",")


###
## NN Model
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
learning_rate = 0.0001 # Learning Rate, could tune
# loss_fn = nn.MSELoss() # l2-norm
loss_fn = nn.L1Loss() # l1-norm
# optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate) # SGD 
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) # Adam

## Training and testing Loops
loss_values = []
test_loss_value = []

for epoch in range(20000): # Learning epoch, could tune

    # Test for plotting
    if epoch % 1000 == 0:
        print(epoch)
    final_test = model(test_trans_data, test_pl_data, test_trpl_data)
    test_loss = loss_fn(final_test, output_test.squeeze(0))
    if epoch % 1000 == 0:
        print("Test loss:", test_loss.item())
    test_loss_value.append(test_loss.item())

    #Training for gradient descent
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


###
## Plot Results
###

## Track the loss of the model over time
step = np.linspace(1,20000,20000)
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