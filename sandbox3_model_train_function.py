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

#reproducibility
torch.manual_seed(42)

#hyperparameters
batch_size = 64
split = [0.9, 0.1]
n_features_in = 1399
n_hidden = 2100
my_learning_rate = 0.001
epochs = 1001

class CustomDataset(Dataset):
    def __init__(self, data_file, transform=None, target_transform=None):
        df = pd.read_csv(data_file, header = 0)
        self.labels = np.asarray(df["sensitive"])
        temp = df.iloc[:, 3:]
        temp.drop(columns=["FO0"], inplace=True)
        self.data = np.asarray(temp, dtype = np.float32) #cols 0,1,2 are labeling stuff
       
        self.transform = transform
        self.target_transform = target_transform
        #x_data and y_data convert np arrays to pytorch tensors
        self.x_data = torch.from_numpy(self.data)
        self.y_data = torch.from_numpy(self.labels)
        
    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        '''Gets a single line of data and its label'''
        return self.x_data[idx], self.y_data[idx]

#training_data = CustomDataset('easy_training_data2.csv')
#testing_data = CustomDataset('easy_testing_data2.csv')
#data = CustomDataset('linear_pipecleaner.csv')
data = CustomDataset('/home/qualcomm_clinic/RTL_dataset/temp_top350.csv')
training_data, testing_data = torch.utils.data.random_split(data, split)

trainingloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
testingloader = torch.utils.data.DataLoader(testing_data, batch_size=batch_size, shuffle=True)

#no need to define custom class, can just use sequential for the not fancy stuff
# to do - probably need to define custom class
'''model = nn.Sequential(
    nn.Linear(n_features_in, n_hidden),
    nn.ReLU(),
    nn.Linear(n_hidden, n_hidden),
    nn.ReLU(),
    nn.Linear(n_hidden, 1)
    #no sigmoid layer because using logits
)'''

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

def model_train(model, trainingloader, testingloader):
    '''function that trains the whole model
        takes in a model and two dataloaders (one for train, one for test)
        returns: (training accuracy, testing accuracy)'''
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

                    test_loss = loss_fn(test_logits, y_test.float()) #feeding logits b/c bcelosswithlogits
                    test_acc = accuracy_fn_bool(y_true=y_test, y_pred=test_pred)

        # Print out what's happening every x epochs
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
        acc_arr.append(acc)
        test_acc_arr.append(test_acc)
    return (acc_arr, test_acc_arr)

print("Starting testing:")
#tr_array = np.arange(0.0002, 0.001, 0.0002)
tr_array = [my_learning_rate]
train = []
test = []
for tr in tr_array:
    print("training rate = " + str(tr))
    model = nn.Sequential(
        nn.Linear(n_features_in, 700),
        nn.ReLU(),
        nn.Linear(700, 350),
        nn.ReLU(),
        nn.Linear(350, 175),
        nn.ReLU(),
        nn.Linear(175, 90),
        nn.ReLU(),
        nn.Linear(90, 45),
        nn.ReLU(),
        nn.Linear(45, 22),
        nn.ReLU(),
        nn.Linear(22, 11),
        nn.ReLU(),
        nn.Linear(11, 5),
        nn.ReLU(),
        nn.Linear(5, 2),
        nn.ReLU(),
        nn.Linear(2, 1)
        #no sigmoid layer because using logits
    )
    (train_acc, test_acc) = model_train(model, trainingloader, testingloader)
    plt.plot(train_acc, label = "training, tr = " + str(tr))
    #plt.plot(test_acc, label = 'testing, tr = ' + str(tr))
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    titlestring = "1000 epochs new crunching"
    plt.title(titlestring)
    plt.savefig("/home/esundheim/feature-extractor/pretty_pictures/" + titlestring + ".png")
  
#to do: use plt.savefig, put into sep folder (on tera), vs code can display images