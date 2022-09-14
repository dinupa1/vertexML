import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import TensorDataset, DataLoader

from vertex import vertexMLP, vertexIN, AverageMeter, draw_loss, draw_test

# load data from .csv file
data = pd.read_csv('data/raw/events-50k.csv')
# print(data.head())

X, y = data[['q1', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'px1', 'py1', 'pz1', 'px2', 'py2', 'pz2']].to_numpy(), data[['vtx', 'vty', 'vtz', 'vpx', 'vpy', 'vpz']].to_numpy()
# print(X.shape, y.shape)


# train, validate, test split
X_trai_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1)
X_train, X_val, y_train, y_val = train_test_split(X_trai_val, y_train_val, test_size=0.22)

print('train shapes {}, {}'.format(X_train.shape, y_train.shape))
print('validate shapes {}, {}'.format(X_val.shape, y_val.shape))
print('test shapes {}, {}'.format(X_test.shape, y_test.shape))

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
net = vertexIN(in_features=13, out_features=6, hidden_dim1=50, hidden_dim2=60)

# print(net)
total_trainable_params = sum(p.numel() for p in net.parameters())
print('total trainable params: {}'.format(total_trainable_params))

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1.0e-14)
epochs = 50

loss = AverageMeter('loss')
train_loss, val_loss = [], []

# Iterate over the dataset <epochs> times
for epoch in range(epochs):
    # Set the model to training mode
    net.train()
    # Reset our meters
    loss.reset()

    # Iterate over batches
    for inputs, targets in train_dataloader:
        # Remove previous gradients
        optimizer.zero_grad()

        # print(inputs.size(), targets.size())

        # Feed forward the input
        outputs = net(inputs)

        # Compute the loss and accuracy
        loss_batch = criterion(outputs, targets)
        loss.update(loss_batch.data)

        # Compute the gradients
        loss_batch.backward()

        # Update parameters
        optimizer.step()

    # Show the current results
    print('Epoch {}, {}'.format(epoch + 1, loss.avg))
    train_loss.append(loss.avg)

    # Validation for each epoch
    net.eval()
    loss.reset()

    for inputs, targets in val_dataloader:
        outputs = net(inputs)

        loss_batch = criterion(outputs, targets)
        loss.update(loss_batch.data)

    val_loss.append(loss.avg)


# Plot the loss
draw_loss(train_loss, val_loss)

# prediction
# y_test = np.ravel(y_test)
y_pred = net(torch.tensor(X_test).unsqueeze(0).float())
y_pred = y_pred.squeeze().detach().numpy()

# print(y_pred)

from sklearn.metrics import r2_score
print('prediction score vtx: {:1.5f}'.format(r2_score(y_test[:, 0], y_pred[:, 0])))
print('prediction score vty: {:1.5f}'.format(r2_score(y_test[:, 1], y_pred[:, 1])))
print('prediction score vtz: {:1.5f}'.format(r2_score(y_test[:, 2], y_pred[:, 2])))
print('prediction score vpx: {:1.5f}'.format(r2_score(y_test[:, 3], y_pred[:, 3])))
print('prediction score vpy: {:1.5f}'.format(r2_score(y_test[:, 4], y_pred[:, 4])))
print('prediction score vpz: {:1.5f}'.format(r2_score(y_test[:, 5], y_pred[:, 5])))

draw_test(y_test[:, 0], y_pred[:, 0], 20, (-5., 5.), 'x [cm]', 'test - preds [cm]', 'vtx-mlp.png')
draw_test(y_test[:, 1], y_pred[:, 1], 20, (-5., 5.), 'y [cm]', 'test - preds [cm]', 'vty-mlp.png')
draw_test(y_test[:, 2], y_pred[:, 2], 250, (-700., 500.), 'z [cm]', 'test - preds [cm]', 'vtz-mlp.png')
draw_test(y_test[:, 3], y_pred[:, 3], 20, (-6., 6.), 'px [GeV/c]', 'test - preds [GeV/c]', 'vpx-mlp.png')
draw_test(y_test[:, 4], y_pred[:, 4], 20, (-6., 6.), 'py [GeV/c]', 'test - preds [GeV/c]', 'vpy-mlp.png')
draw_test(y_test[:, 5], y_pred[:, 5], 20, (10., 100.), 'pz [GeV/c]', 'test - preds [GeV/c]', 'vpz-mlp.png')