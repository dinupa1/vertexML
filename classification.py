"""
author Dinupa Nawarathne
email dinupa3@gmail.com
date 10-07-2022


Classification of the z vertex position of the reconstructed single muon track.
We use the MLPClassifier in sklearn library and pytorch for this task.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
function to add labels
position                 | label
------------------------------------------
-700.0 <= z < -304.0    | collimeter
-304.0 <= z <= -296.0   | target
-296.0 < z < 0.0        | other
0.0 <= z <= 500.0       | beam_dump
"""

def add_labels(z: float)->int:
    if -800.0 < z and z < -305.0:
        return 0
    if -305.0 < z and z < -295.0:
        return 1
    if -295.0 < z and z < -1.0:
        return 2
    if -1.0 < z and z < 500.0:
        return 3
    else:
        return -99

hep.style.use(hep.style.ROOT)

df = pd.read_csv('events-200k.csv')
df['hot_id']  = df['vtz'].apply(add_labels)
# print(df[df['hot_id']=="no_pos"]['vtz'])

# split train and target
X, y_label = df[['q1', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'px1', 'py1', 'pz1', 'px2', 'py2', 'pz2']].to_numpy(), df['hot_id'].to_numpy()
# print(y_label)


# plot labels
# plt.hist(y_label)
# plt.yscale('log')
# plt.show()

# one hot encoding
ohe = OneHotEncoder(sparse=False)
y_true = ohe.fit_transform(y_label.reshape(-1, 1))
# print(ohe.categories_)


# train, validate test split
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_true, test_size=0.2)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_val, y_train_val, test_size=0.25)

# scale the train, validate and test inputs
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

print(X_test.shape)
print(y_test.shape)

"""
Classification using sklearn MLPClassifier
info https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
"""

# clf = MLPClassifier(max_iter=300)
# clf.fit(X_train, y_train)
# score = clf.score(X_test, y_test)
# print("prediction score = {}".format(score))
#
# y_pred = clf.predict(X_test)
# y_pred_deco = np.argmax(y_pred, axis=1)
# y_test_deco = np.argmax(y_test, axis=1)
#
# plt.hist(y_test_deco, histtype='step', label='test')
# plt.hist(y_pred_deco, histtype='step', label='pred')
# plt.legend()
# plt.yscale('log')
# plt.show()

"""
using pytorch we build a neural network for regression
"""

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

class vertexTagger(torch.nn.Module):
    def __init__(self, in_features: int = 13, out_features: int =4, hidden_dim: int = 32):
        super(vertexTagger, self).__init__()
        self.fc1 = torch.nn.Linear(in_features, hidden_dim, bias=True)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc4 = torch.nn.Linear(hidden_dim, out_features, bias=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=-1)
        return x



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':1.5f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{avg' + self.fmt + '} ({name})'
        return fmtstr.format(**self.__dict__)


batch_size = 1024

train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(torch.Tensor(X_valid), torch.Tensor(y_valid))
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

net = vertexTagger(hidden_dim=32)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
epochs = 20


# train the neural networks
import sys

from tqdm.notebook import trange

acc, loss = AverageMeter('Accuracy'), AverageMeter('Loss')
train_loss, val_acc, val_loss = [], [], []

# Iterate over the dataset <epochs> times
for epoch in range(epochs):

    # Set the model to training mode
    net.train()
    # Reset our meters
    loss.reset()
    acc.reset()

    # This is just here because it's pretty
    tr = trange(len(train_dataloader), file=sys.stdout)

    # Iterate over batches
    for inputs, targets in train_dataloader:
        # Remove previous gradients
        optimizer.zero_grad()

        # Feed forward the input
        outputs = net(inputs)

        # Compute the loss and accuracy
        loss_batch = criterion(outputs, targets)
        loss.update(loss_batch.data)

        preds = torch.argmax(outputs, dim=-1)
        accuracy = (torch.argmax(targets, dim=-1) == preds).sum() / len(targets)
        acc.update(accuracy.data)

        # Show the current results
        tr.set_description('Epoch {}, {}, {}'.format(epoch + 1, loss, acc))
        tr.update(1)

        # Compute the gradients
        loss_batch.backward()

        # Update parameters
        optimizer.step()

    train_loss.append(loss.avg)

    # Validation for each epoch
    net.eval()
    loss.reset()
    acc.reset()

    tr = trange(len(val_dataloader), file=sys.stdout)

    for inputs, targets in val_dataloader:
        outputs = net(inputs)

        loss_batch = criterion(outputs, targets)
        loss.update(loss_batch.data)

        preds = torch.argmax(outputs, dim=-1)
        accuracy = (torch.argmax(targets, dim=-1) == preds).sum() / len(targets)
        acc.update(accuracy.data)

        tr.set_description('Validation, {}, {}'.format(loss, acc))
        tr.update(1)

    val_loss.append(loss.avg)
    val_acc.append(acc.avg)

def draw_loss(data_train, data_val, data_acc, label="Loss"):
    """Plots the training and validation loss"""

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Epoch", horizontalalignment='right', x=1.0)
    ax1.set_ylabel("Loss", horizontalalignment='right', y=1.0)
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.plot(data_train,
             color='red',
             label='Training loss')
    ax1.plot(data_val,
             color='blue',
             label='Validation loss')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.plot(data_acc,
             color='green',
             label='Accuracy')
    ax1.legend(loc='lower left')
    ax2.legend(loc='upper left')
    plt.show()

draw_loss(train_loss, val_loss, val_acc)


from sklearn.metrics import accuracy_score

y_pred = net(torch.tensor(X_test).unsqueeze(0).float())

print("Accuracy for the test set: {0:.2f}".format(
    accuracy_score(
        np.argmax(y_test, axis=1),
        torch.argmax(y_pred, dim=-1).squeeze().numpy())
))

plt.hist(torch.argmax(y_pred, dim=-1).squeeze().numpy())
plt.show()
# plt.hist(np.argmax(y_test, axis=1), histtype='step', label='test')
# plt.hist(torch.argmax(y_pred, dim=-1).squeeze().numpy(), histtype='step', label='pred')
# plt.legend()
# plt.yscale('log')
# plt.show()

# from sklearn.metrics import roc_curve, auc
#
# def plot_roc(y_test, y_pred, labels):
#     for x, label in enumerate(labels):
#         fpr, tpr, _ = roc_curve(y_test[:, x], y_pred[:, x])
#         plt.plot(tpr, fpr, label='{0} tagger, AUC = {1:.1f}'.format(label, auc(fpr, tpr) * 100.), linestyle='-')
#     plt.semilogy()
#     plt.xlabel("Signal Efficiency")
#     plt.ylabel("Background Efficiency")
#     plt.ylim(0.001, 1)
#     plt.grid(True)
#     plt.legend(loc='upper left')
#     plt.show()
#
# plt.figure(figsize=(5, 5))
# plot_roc(y_test, y_pred.squeeze().detach().numpy(), ohe.categories_)