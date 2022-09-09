from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import mplhep as hep

plt.style.use(hep.style.ROOT)

epochs = 1000

data = pd.read_csv('data/raw/events-20k.csv')

# print(data.head())

X, y = data[['q1', 'x1', 'y1', 'z1', 'px1', 'py1', 'pz1', 'x2', 'y2', 'z2', 'px2', 'py2', 'pz2']].to_numpy(), data[['vtx', 'vty', 'vtz']].to_numpy()

# print(X.shape)
# print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25)

# print(X_train.shape)
# print(X_val.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_val.shape)
# print(y_test.shape)

# scale the inputs
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

mlp = MLPRegressor(hidden_layer_sizes=(512, 256, 256, 64, 64),
                   random_state=1,
                   max_iter=epochs,
                   solver='adam',
                   validation_fraction=0.25,
                   batch_size=128)


# print(mlp)

mlp.fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(mlp.loss_curve_, label='train')
ax.set_xlabel('epochs')
ax.set_ylabel('mse')
# plt.plot(mlp.validation_scores_, label='validation')
fig.tight_layout()
plt.savefig('mlp-loss-curve.png')

y_predict = mlp.predict(X_test)

score = mlp.score(X_test, y_test)

print('prediction score = {}'.format(score))

# print(len(mlp.loss_curve_))


fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].hist(y_test[:, 0], bins=20, range=(-5.0, 5.0), histtype='step', label='true')
ax[0].hist(y_predict[:, 0], bins=20, range=(-5.0, 5.0), histtype='step', label='prediction')
ax[0].set_xlabel('z [cm]')
ax[0].set_ylabel('counts')
ax[0].legend()

ax[1].hist(y_test[:, 0]-y_predict[:, 0], bins=20)
ax[1].set_xlabel('z_true - z_pred [cm]')
fig.tight_layout()
plt.savefig('mlp-vtx.png')

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].hist(y_test[:, 1], bins=20, range=(-5.0, 5.0), histtype='step', label='true')
ax[0].hist(y_predict[:, 1], bins=20, range=(-5.0, 5.0), histtype='step', label='prediction')
ax[0].set_xlabel('z [cm]')
ax[0].set_ylabel('counts')
ax[0].legend()

ax[1].hist(y_test[:, 1]-y_predict[:, 1], bins=20)
ax[1].set_xlabel('z_true - z_pred [cm]')
fig.tight_layout()
plt.savefig('mlp-vty.png')


fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].hist(y_test[:, 2], bins=20, range=(-800.0, 200.0), histtype='step', label='true')
ax[0].hist(y_predict[:, 2], bins=20, range=(-800.0, 200.0), histtype='step', label='prediction')
ax[0].set_xlabel('z [cm]')
ax[0].set_ylabel('counts')
ax[0].legend()

ax[1].hist(y_test[:, 2]-y_predict[:, 2], bins=20)
ax[1].set_xlabel('z_true - z_pred [cm]')
fig.tight_layout()
plt.savefig('mlp-vtz.png')