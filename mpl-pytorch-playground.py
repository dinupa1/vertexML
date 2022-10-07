import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt

from vertex import vertexMLP, vertexIN, AverageMeter, draw_loss, draw_test, draw_weights, draw_cuts, cuts

ohe = OneHotEncoder()

# load data from .csv file
data = pd.read_csv('data/raw/events-200k.csv')
data['label'] = data['vtz'].apply(cuts)
# print(data.head())

# print(data.loc[data['label']=='target'])


hot_id = ohe.fit_transform(data[['label']])
hot_id = hot_id.toarray()

# print(hot_id)

data[ohe.categories_[0]] = hot_id.tolist()
# print(data.head())

X, y = data[['q1', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'px1', 'py1', 'pz1', 'px2', 'py2', 'pz2']].to_numpy(),\
       data[['vtx', 'vty', 'vtz', 'vpx', 'vpy', 'vpz', 'collimeter', 'target', 'other', 'beam_dump']].to_numpy()
# print(X.shape, y.shape)


# train, validate, test split
X_trai_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_trai_val, y_train_val, test_size=0.25)


# plt.hist(y_train[:, 6:])
# plt.yscale('log')
# plt.show()

print('train shapes {}, {}'.format(X_train.shape, y_train.shape))
print('validate shapes {}, {}'.format(X_val.shape, y_val.shape))
print('test shapes {}, {}'.format(X_test.shape, y_test.shape))

# Plot train target cuts
# print(y_test[:, 2])
# draw_cuts(y_test[:, 2], y_test[:, 3], 'vtz [cm]', 'vpx [GeV/c]', 'vtz_vpx')
# draw_cuts(y_test[:, 2], y_test[:, 4], 'vtz [cm]', 'vpy [GeV/c]', 'vtz_vpy')
# draw_cuts(y_test[:, 2], y_test[:, 5], 'vtz [cm]', 'vpz [GeV/c]', 'vtz_vpz')

# scale the input data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# convert to torch data set
batch_size = 64
train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# construct the model
net = vertexIN(hidden_dim1=60, hidden_dim2=80)

# print(net)
total_trainable_params = sum(p.numel() for p in net.parameters())
print('total trainable params: {}'.format(total_trainable_params))

# criterion_clf = torch.nn.CrossEntropyLoss()
criterion_clf = torch.nn.CrossEntropyLoss()
criterion_reg = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
# scheduler = StepLR(optimizer, step_size=5, gamma=0.005)
epochs = 1

loss_clf_meter, loss_reg_meter, loss_total_meter = AverageMeter('loss_clf'), AverageMeter('loss_reg'), AverageMeter('loss_total')
train_loss_clf, train_loss_reg, train_loss_total, val_loss_clf, val_loss_reg, val_loss_total = [], [], [], [], [], []

# Iterate over the dataset <epochs> times
for epoch in range(epochs):
    # Set the model to training mode
    net.train()
    # Reset our meters
    loss_clf_meter.reset()
    loss_reg_meter.reset()
    loss_total_meter.reset()

    # Iterate over batches
    for inputs, targets in train_dataloader:
        # Remove previous gradients
        optimizer.zero_grad()

        # print(inputs.size(), targets[:, 6:].size())
        # print(targets[:, 6:])

        # Feed forward the input
        pred_id, pred_mom = net(inputs)

        # print(pred_id, pred_mom)

        # print(pred_id.size(), pred_mom.size())

        # Compute the loss and accuracy
        loss_batch_clf = criterion_clf(pred_id, targets[:, 6:])
        loss_batch_reg = criterion_reg(pred_mom, targets[:, :6])

        loss_batch_total = loss_batch_clf + loss_batch_reg

        loss_clf_meter.update(loss_batch_clf.data)
        loss_reg_meter.update(loss_batch_reg.data)
        loss_total_meter.update(loss_batch_total.data)

        # Compute the gradients
        loss_batch_total.backward()

        # Update parameters
        optimizer.step()

    # Show the current results
    print('Epoch {}, {}'.format(epoch + 1, loss_clf_meter.avg))
    print('Epoch {}, {}'.format(epoch + 1, loss_reg_meter.avg))
    print('Epoch {}, {}'.format(epoch + 1, loss_total_meter.avg))
    train_loss_clf.append(loss_clf_meter.avg)
    train_loss_reg.append(loss_reg_meter.avg)
    train_loss_total.append(loss_total_meter.avg)

    # Validation for each epoch
    net.eval()

    loss_clf_meter.reset()
    loss_reg_meter.reset()
    loss_total_meter.reset()

    for inputs, targets in val_dataloader:
        pred_id, pred_mom = net(inputs)

        loss_batch_clf = criterion_clf(pred_id, targets[:, 6:])
        loss_batch_reg = criterion_reg(pred_mom, targets[:, :6])

        loss_batch_total = loss_batch_clf + loss_batch_reg

        loss_clf_meter.update(loss_batch_clf.data)
        loss_reg_meter.update(loss_batch_reg.data)
        loss_total_meter.update(loss_batch_total.data)

    val_loss_clf.append(loss_clf_meter.avg)
    val_loss_reg.append(loss_reg_meter.avg)
    val_loss_total.append(loss_total_meter.avg)

    # scheduler.step()


# Plot the loss
draw_loss(train_loss_clf, train_loss_reg, val_loss_clf, val_loss_reg)

# Plot the weights in the hidden layers
# draw_weights(net.dnn1[2].weight, 'dnn1_2.png')
# draw_weights(net.dnn1[4].weight, 'dnn1_4.png')
# draw_weights(net.dnn1[6].weight, 'dnn1_6.png')
# draw_weights(net.dnn2[2].weight, 'dnn2_2.png')

# Make prediction
y_id, y_pred = net(torch.tensor(X_test).float())
y_pred = y_pred.squeeze().detach().numpy()

# print(y_pred)

# Prediction score
from sklearn.metrics import r2_score
print('prediction score vtx: {:1.5f}'.format(r2_score(y_test[:, 0], y_pred[:, 0])))
print('prediction score vty: {:1.5f}'.format(r2_score(y_test[:, 1], y_pred[:, 1])))
print('prediction score vtz: {:1.5f}'.format(r2_score(y_test[:, 2], y_pred[:, 2])))
print('prediction score vpx: {:1.5f}'.format(r2_score(y_test[:, 3], y_pred[:, 3])))
print('prediction score vpy: {:1.5f}'.format(r2_score(y_test[:, 4], y_pred[:, 4])))
print('prediction score vpz: {:1.5f}'.format(r2_score(y_test[:, 5], y_pred[:, 5])))


# Plot the results
draw_test(y_test[:, 0], y_pred[:, 0], 20, (-5., 5.), 'x [cm]', 'test - preds [cm]', 'vtx-mlp.png')
draw_test(y_test[:, 1], y_pred[:, 1], 20, (-5., 5.), 'y [cm]', 'test - preds [cm]', 'vty-mlp.png')
draw_test(y_test[:, 2], y_pred[:, 2], 50, (-700., 500.), 'z [cm]', 'test - preds [cm]', 'vtz-mlp.png')
draw_test(y_test[:, 3], y_pred[:, 3], 20, (-6., 6.), 'px [GeV/c]', 'test - preds [GeV/c]', 'vpx-mlp.png')
draw_test(y_test[:, 4], y_pred[:, 4], 20, (-6., 6.), 'py [GeV/c]', 'test - preds [GeV/c]', 'vpy-mlp.png')
draw_test(y_test[:, 5], y_pred[:, 5], 20, (10., 100.), 'pz [GeV/c]', 'test - preds [GeV/c]', 'vpz-mlp.png')