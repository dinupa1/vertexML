import torch
import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt
import mplhep as hep

plt.style.use(hep.style.ROOT)


# MLP neural network
class vertexMLP(torch.nn.Module):
    def __init__(self, in_features: int = 13, out_features: int = 3, hidden_dim: int = 1000):
        super(vertexMLP, self).__init__()
        # Let's define our layers here.
        self.fc1 = torch.nn.Linear(in_features, hidden_dim, bias=True)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc4 = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc5 = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc6 = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc7 = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc8 = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc9 = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc10 = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc11 = torch.nn.Linear(hidden_dim, out_features, bias=True)

    def forward(self, x):
        # Let's define out forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = F.relu(self.fc10(x))
        x = self.fc11(x)


class MLP1(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_dim: int):
        super(MLP1, self).__init__()
        self.in_layer1 = torch.nn.Linear(in_features, hidden_dim)
        self.hidden_layer11 = torch.nn.Linear(hidden_dim, hidden_dim)

        # self.batch_n11 = torch.nn.BatchNorm1d(hidden_dim)
        # self.dropout11 = torch.nn.Dropout(0.2)

        self.hidden_layer12 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer13 = torch.nn.Linear(hidden_dim, hidden_dim)

        # self.batch_n12 = torch.nn.BatchNorm1d(hidden_dim)
        # self.dropout12 = torch.nn.Dropout(0.2)

        self.out_layer1 = torch.nn.Linear(hidden_dim, out_features)
        # self.dropout12 = torch.nn.Dropout(0.25)

        # Initializations
        torch.nn.init.xavier_normal_(self.in_layer1.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_normal_(self.hidden_layer11.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_normal_(self.hidden_layer12.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_normal_(self.hidden_layer13.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_normal_(self.out_layer1.weight, gain=torch.nn.init.calculate_gain('relu'))

        self.relu1 = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.in_layer1(x))
        x = F.relu(self.hidden_layer11(x))

        # x = F.relu(self.batch_n11(x))
        # x = self.dropout11(x)

        x = F.relu(self.hidden_layer12(x))
        x = F.relu(self.hidden_layer13(x))

        # x = F.relu(self.batch_n12(x))
        # x = self.dropout12(x)

        x = self.out_layer1(x)
        # x = self.sigmoid(self.out_layer1(x))
        return x


class MLP2(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_dim: int):
        super(MLP2, self).__init__()
        self.in_layer2 = torch.nn.Linear(in_features, hidden_dim)
        self.hidden_layer21 = torch.nn.Linear(hidden_dim, hidden_dim)

        # self.batch_n21 = torch.nn.BatchNorm1d(hidden_dim)
        # self.dropout21 = torch.nn.Dropout(0.2)

        self.hidden_layer22 = torch.nn.Linear(hidden_dim, hidden_dim)
        # self.hidden_layer23 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.out_layer2 = torch.nn.Linear(hidden_dim, out_features)

        # Initializations
        torch.nn.init.xavier_normal_(self.in_layer2.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_normal_(self.hidden_layer21.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_normal_(self.hidden_layer22.weight, gain=torch.nn.init.calculate_gain('relu'))
        # torch.nn.init.xavier_normal(self.hidden_layer13)
        torch.nn.init.xavier_normal_(self.out_layer2.weight, gain=torch.nn.init.calculate_gain('relu'))

        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu2(self.in_layer2(x))
        x = F.relu(self.hidden_layer21(x))
        x = F.relu(self.hidden_layer22(x))

        # x = F.relu(self.batch_n21(x))
        # x = self.dropout21(x)

        x = self.out_layer2(x)

        return x


# interaction neural network
class vertexIN(torch.nn.Module):
    def __init__(self, in_features: int = 13, out_features: int = 7, hidden_dim1: int = 300, hidden_dim2: int = 350):
        super(vertexIN, self).__init__()

        self.dnn1 = MLP1(in_features, 5, hidden_dim1)
        self.dnn2 = MLP2(in_features+5, 6, hidden_dim2)

    def forward(self, x):
        pred_id = self.dnn1(x)
        # print(pred_id)
        pred_mom = self.dnn2(torch.cat([x, pred_id], axis=-1))
        # preds = torch.cat([pred_pos, pred_mom], axis=-1)

        return pred_id, pred_mom


# custom lass to store the loss
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


# Let's draw loss for the training and validation
def draw_loss(data_train_clf: list, data_train_reg: str, data_val_clf: list, data_val_reg: str, label: str = "Loss"):
    """Plots the training and validation loss"""

    fig, ax = plt.subplots()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.plot(data_train_clf, color='red', label='Training clf-loss')
    ax.plot(data_train_reg, color='blue', label='Training reg-loss')
    ax.plot(data_val_clf, color='orange', label='Validation clf-loss')
    ax.plot(data_val_reg, color='orange', label='Validation clf-loss')
    ax.legend(loc='upper right', borderpad=0.1, markerscale=0.1, fontsize='small')
    plt.savefig('loss-mlp.png')
    # plt.show()


# Let's draw the predictions
def draw_test(y_test: np.ndarray, y_pred: np.ndarray, bins: int, range: tuple, label1: str, label2: str, fig_name: str):
    """Plot the predictions"""

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].hist(y_test, bins=bins, range=range, histtype='step', label='test')
    ax[0].hist(y_pred, bins=bins, range=range, histtype='step', label='prediction')
    ax[0].set_xlabel(label1)
    ax[0].set_ylabel('count')
    ax[0].set_yscale('log')
    ax[0].legend(loc='upper right', borderpad=0.1, markerscale=0.1, fontsize='xx-small')

    ax[1].hist(y_test - y_pred, bins=bins)
    ax[1].set_xlabel(label2)
    ax[1].set_yscale('log')
    fig.tight_layout()
    plt.savefig(fig_name)
    # plt.show()

def draw_weights(weights: torch.Tensor, save_name: str):

    array = weights.detach().numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hist(np.ravel(array), bins=20)
    ax.set_xlabel('weights')
    ax.set_ylabel('count')
    fig.tight_layout()
    plt.savefig(save_name)
    # plt.show()
def draw_cuts(x, y, xlabel, ylabel, save_name):

    fig, ax = plt.subplots(figsize=(5, 5))
    h = ax.hist2d(x, y, bins=[50, 20])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(h[3], ax=ax)
    fig.tight_layout()
    plt.savefig(save_name+'.png')


def cuts(x):
    if -700.0 <= x and x < -304.0:
        return "collimeter"
    if -304.0 <= x and x <= -296.0:
        return "target"
    if -296.0 < x and x < 0.0:
        return "other"
    if 0.0 <= x and x <= 500.0:
        return "beam_dump"