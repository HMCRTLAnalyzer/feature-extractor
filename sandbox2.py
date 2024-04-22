import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchdata.datapipes as dp
import os

#reporduciblity
torch.manual_seed(42)

#hyperparameters
batch_size = 64
split = [0.9, 0.1]
n_hidden = 150
my_learning_rate = 0.001
epochs = 100

class CustomDataset(Dataset):
    def __init__(self, data_file, transform=None, target_transform=None):
        df = pd.read_csv(data_file, header = 0)
        self.data = df[["feature0", "feature1", "feature2", "feature3", "feature4"]]
        
        #labels are the outputs: in this dummy data, 1 or 0
        self.labels = np.asarray(df["output"])
        self.transform = transform
        self.target_transform = target_transform
        self.x_data = torch.tensor(np.asarray(self.data), dtype = torch.float32)
        self.y_data = torch.tensor(self.labels)
        

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        '''Gets a single line of data and its label'''
        return self.x_data[idx], self.y_data[idx]

#training_data = CustomDataset('easy_training_data2.csv')
#testing_data = CustomDataset('easy_testing_data2.csv')
data = CustomDataset('linear_pipecleaner.csv')
training_data, testing_data = torch.utils.data.random_split(data, split)


trainingloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
testingloader = torch.utils.data.DataLoader(testing_data, batch_size=batch_size, shuffle=True)

#no need to define custom class, can just use sequential for the not fancy stuff

model = nn.Sequential(
    nn.Linear(5, n_hidden),
    nn.ReLU(),
    nn.Linear(n_hidden, n_hidden),
    nn.ReLU(),
    nn.Linear(n_hidden, 1)
    #no sigmoid layer because using logits
)

def accuracy_fn(y_true, y_pred):
    '''evaluation metric for model'''
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_true)) * 100 
    return acc

def accuracy_fn_bool(y_true, y_pred):
    '''evaluation metric for model that plays nicely with data formats'''
    acc = 0
    correct = 0
    for n in range(len(y_true)):
       # print(y_true[n])
       # print(y_pred[n])
        if (y_true[n].bool() == y_pred[n].bool()):
            correct += 1
    acc = (correct / len(y_true)) * 100 
    return acc

#binary classifier -sens or not sens
loss_fn = nn.BCEWithLogitsLoss()
bce = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=my_learning_rate)

#training loop
acc_arr = []
test_acc_arr = []
for epoch in range(epochs):
    for X, y in trainingloader:

        model.train()
        pred_logits = model(X)
        pred_logits.squeeze(dim=1) #get rid of extra dimension

        pred_probs = torch.sigmoid(pred_logits)
        pred_labels = torch.round(pred_probs)

        '''
        print("-------------------------logits")
        print(pred_logits[:10])
        test_logits = pred_logits[:10]
        
        print("-------------------------pred probs")
        print(pred_probs[:10])
        test_probs = torch.sigmoid(test_logits)

        print("-------------------------y labels")
        print(y[:10])
        
        print("-------------------------predicted labels")
        print(pred_labels[:10])
        '''
        
        pred_logits_squeezed = pred_logits.squeeze(dim = 1)
        pred_probs_squeezed = pred_probs.squeeze(dim = 1)
        pred_labels_squeezed = pred_labels.squeeze(dim = 1)

        #loss_fn is an instance of the BCELosswithLogits class
        #syntax is criterion = (output, target)
        loss = loss_fn(pred_logits_squeezed, y.float()) 
        acc = accuracy_fn_bool(y_true=y, y_pred=pred_labels_squeezed) 
        #acc_arr.append(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### Testing
        model.eval()
        with torch.inference_mode():
            # 1. Forward pass
            #test_acc_arr = []
            for X_test, y_test in testingloader:
                test_logits = model(X_test).squeeze() 
                test_pred = torch.round(torch.sigmoid(test_logits))

                #print(f"\nFirst 10 predictions:\n{test_pred[:10]}")
                #print(f"\nFirst 10 test labels:\n{y_test[:10]}")

                # 2. Caculate loss/accuracy
                test_loss = loss_fn(test_logits, y_test.float()) #feeding logits b/c bcelosswithlogits
                test_acc = accuracy_fn_bool(y_true=y_test, y_pred=test_pred)

    # Print out what's happening every x epochs
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
        acc_arr.append(acc)
        test_acc_arr.append(test_acc)
