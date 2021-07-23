#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 00:27:22 2021

@author: jyoti
"""

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as f
from torch import nn
import scipy.io as sio
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
from multiprocessing import cpu_count
import torch.nn.functional as F
import torch.optim as optim



torch.cuda.empty_cache()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#load the ECG file
src_dataset = sio.loadmat('/home/jyoti/MasterThesis/DataFall/DataFall/implementation/implementation/ECG2(withDA)')
#load the labels
label=pd.read_csv('/home/jyoti/MasterThesis/DataFall/DataFall/implementation/implementation/label.csv',header=None)

#print(src_dataset)
testdata = src_dataset['ECG'] # use the key for data here
X=testdata['Data']
X=np.array(X[0])
X = np.vstack(X[:,]).astype(np.float)

label=np.array(label, dtype=object)
label=np.array(label)
label= np.vstack(label[:,]).astype(np.float)

df = pd.DataFrame(data =X)
label = pd.DataFrame(data = label)
df1 = pd.merge(df,label, left_index = True, right_index=True)
a = df1.loc[df1['0_y']== 0,:]
b = df1.loc[df1['0_y']== 1,:]
c = df1.loc[df1['0_y']== 2,:]
#take 250 rows from each of the section
a = a[1:250]
b = b[1:250]
c = c[1:250]
a = a.reset_index(drop = True)
b =b.reset_index(drop = True)
c = c.reset_index(drop = True)

df2 = pd.concat([a,b])
df2 = pd.concat([df2,c])
df2 = df2.reset_index(drop = True)
label_1 = df2['0_y']
del df2['0_y']
label_1 = label_1.to_numpy()
Y = torch.tensor(label_1)
#Y = torch.tensor(label)
X = df2.to_numpy()
X=torch.from_numpy(X)


def create_datasets(X, y, test_size=0.2, time_dim_first=False):
    enc = LabelEncoder()
    y_enc = enc.fit_transform(y)
    X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, stratify=Y)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=1)  # 0.25 x 0.8 = 0.2
    print(X_train.shape)
    print(X_valid.shape)
    print(y_train.shape)
    y_train=torch.squeeze(y_train)
    print(y_test.shape)
    y_valid=torch.squeeze(y_valid)
    y_test=torch.squeeze(y_test)

    X_train, X_valid, X_test = [torch.tensor(arr, dtype=torch.float32) for arr in (X_train, X_valid,X_test)]
    y_train, y_valid,y_test = [torch.tensor(arr, dtype=torch.long) for arr in (y_train, y_valid,X_test)]
    print(X_train.shape)
    print(X_valid.shape)
    print(X_test.shape)
    train_ds = TensorDataset(X_train, y_train)
    valid_ds = TensorDataset(X_valid, y_valid)
    tst_ds=TensorDataset(X_test,y_test)
    return train_ds, valid_ds,tst_ds, enc

def create_loaders(train_ds, valid_ds,tst_ds, bs=512, jobs=0):
    train_dl = DataLoader(train_ds, bs, shuffle=True, num_workers=jobs)
    valid_dl = DataLoader(valid_ds, bs, shuffle=False, num_workers=jobs)
    tst_dl = DataLoader(tst_ds, bs, shuffle=False, num_workers=jobs)
    return train_dl, valid_dl,tst_ds

def accuracy(output, target):
    return (output.argmax(dim=1) == target).float().mean().item()

class LSTMClassifier(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        #self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        #self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_size = None
        self.hidden = None
        #self.log_softmax = None

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        #x = torch.unsqueeze(x, 2)
        print(x.size())
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.to(device) for t in (h0, c0)]


trn_ds, val_ds, tst_data, enc = create_datasets(X, Y)

trn_ds, val_ds, tst_data, enc = create_datasets(X, Y)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
bs =50#128
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Creating data loaders with batch size: {bs}')
trn_dl, val_dl,tst_data = create_loaders(trn_ds, val_ds,tst_data, bs, jobs=cpu_count())
input_dim = 1
hidden_dim = 500
layer_dim = 1
output_dim = 1
seq_dim = 650
lr = 0.01
n_epochs = 101
iterations_per_epoch = len(trn_dl)
best_acc = 0
patience, trials = 100, 0
#model = LSTM()
model = LSTMClassifier(input_dim,hidden_dim,layer_dim,output_dim)

model = model.to(device)

loss_function = nn.CrossEntropyLoss()
#opt = torch.optim.Adam(model.parameters(), lr=0.001)
#criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(1, n_epochs + 1):

    for i, (x_batch, y_batch) in enumerate(trn_dl):
        model.train()
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        print('shape of the input batch')
        print(x_batch.shape)
        opt.zero_grad()
        x_batch=torch.unsqueeze(x_batch,2)
        print(x_batch.shape)
        out = model(x_batch)
        y_batch=torch.unsqueeze(y_batch,0)
        #print(out.shape)
        print('NOW')
        print(y_batch.dtype)
        y_batch = y_batch.to(torch.float32)

        out = out.to(torch.float32)
        out=torch.transpose(out,1,0)
        loss = loss_function(out, torch.max(y_batch, 1)[1])
        #(out, y_batch)
        #targets = targets.to(torch.float32)
        loss.backward()
        opt.step()

    model.eval()
    correct, total = 0, 0
    #problem with the x_val size and type here
    for x_val, y_val in val_dl:


        x_val, y_val = [t for t in (x_val, y_val)]
        x_val = x_val.to(device)
        y_val = y_val.to(device)
        x_val=torch.unsqueeze(x_val,2)
        print(x_val.shape)
        out = model(x_val)
        preds = F.log_softmax(out, dim=1).argmax(dim=1)
        total += y_val.size(0)
        correct += (preds == y_val).sum().item()

    acc = correct / total

    if epoch % 5 == 0:
        print(f'Epoch: {epoch:3d}. Loss: {loss.item():.4f}. Acc.: {acc:2.2%}')

    if acc > best_acc:
        trials = 0
        best_acc = acc
        torch.save(model.state_dict(), 'best.pth')
        print(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
    else:
        trials += 1
        if trials >= patience:
            print(f'Early stopping on epoch {epoch}')
            break
