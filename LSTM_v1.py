#from https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51
#frank odom
#modified by FSB
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
#load the ECG file
src_dataset = sio.loadmat('/Users/fatima/PycharmProjects/LSTM/venv/ECG2(withDA).mat')
#load the labels
label=pd.read_csv('/Users/fatima/PycharmProjects/LSTM/venv/label.csv',header=None)

#print(src_dataset)
testdata = src_dataset['ECG'] # use the key for data here
#testlabel = src_dataset['label'] # use the key for target here
data=testdata['Data']
#label=testdata['label']
data=np.array(data[0])
data = np.vstack(data[:,]).astype(np.float)
print(data.shape)
#convert to torch
X=torch.from_numpy(data)

label=np.array(label,dtype=object)
#label=label.transpose()
print(label.shape)
label=np.array(label)
label = np.vstack(label[:,]).astype(np.float)

X=torch.from_numpy(data)
trg = torch.tensor(label)
Y = trg.float()
print(Y.shape)
print(X.shape)

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
    #y_train, y_valid = [torch.tensor(arr, dtype=torch.long) for arr in (y_train, y_valid)]

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

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        #x = torch.unsqueeze(x, 2)
        print(x.size())
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    #def forward(self, x):
        # Initializing hidden state for first input with zeros
     #   h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initializing cell state for first input with zeros
      #  c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
       # out, (hn, cn) = self.lstm(x, (h0, c0))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        #out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        #out = self.fc(out)
        #print('inside')
        #x=torch.transpose(x,0,1)
        #x=torch.tensor(x)
        #x=torch.unsqueeze(x,2)
        #print(x.size())
        #h0, c0 = self.init_hidden(x)
        #out, (hn, cn) = self.rnn(x, (h0, c0))
        #out = self.fc(out[:, -1, :])
        #print('AFTER')
        #return out

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)

        #h0 = torch.zeros(x.size(0), self.hidden_dim,self.layer_dim)
        #c0 = torch.zeros( x.size(0), self.hidden_dim,self.layer_dim)
        print('hidden layer')
        print(h0.shape)
        print(x.size(0))
        print(layer_dim)
        return [t.to(device) for t in (h0, c0)]

trn_ds, val_ds, tst_data, enc = create_datasets(X, Y)
#print(tst_data.shape)
bs =100#128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Creating data loaders with batch size: {bs}')
trn_dl, val_dl,tst_data = create_loaders(trn_ds, val_ds,tst_data, bs, jobs=cpu_count())
input_dim = 1
hidden_dim = 500
layer_dim = 1
output_dim = 1
seq_dim = 4000
lr = 0.01
n_epochs = 101
iterations_per_epoch = len(trn_dl)
best_acc = 0
patience, trials = 100, 0
#model = LSTM()
model = LSTMClassifier(input_dim,hidden_dim,layer_dim,output_dim)

model = model.to(device)

loss_function = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
#criterion = nn.CrossEntropyLoss()
opt = torch.optim.RMSprop(model.parameters(), lr=lr)

print('Start model training')



for epoch in range(1, n_epochs + 1):

    for i, (x_batch, y_batch) in enumerate(trn_dl):
        model.train()
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        #sched.step()

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

model.load_state_dict(torch.load('best.pth'))
model.eval()

test_dl = DataLoader(tst_data, batch_size=64, shuffle=False)
test = []
print('Predicting on test dataset')
for batch, _ in tst_data:
    #test_dl:
    #batch = batch.permute(0, 2, 1)
    batch=batch.to(device)
    print(batch.shape)
    out = model(device)
    y_hat = F.log_softmax(out, dim=1).argmax(dim=1)
    test += y_hat.tolist()