import torch
import torch.nn.functional as F

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


# interaction neural network
class vertexIN(torch.nn.Module):
    def __init__(self, in_features: int = 13, out_features: int = 3, hidden_dim1: int = 300, hidden_dim2: int = 350):
        super(vertexIN, self).__init__()

        self.dnn1 = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_dim1, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim1, hidden_dim1, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim1, hidden_dim1, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim1, hidden_dim1, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim1, hidden_dim1, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim1, 3, bias=True)
        )

        self.dnn2 = torch.nn.Sequential(
            torch.nn.Linear(in_features + 3, hidden_dim2, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim2, hidden_dim2, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim2, hidden_dim2, bias=True),
            torch.nn.ReLU(),
            # torch.nn.Linear(hidden_dim2, hidden_dim2, bias=True),
            # torch.nn.ReLU(),
            # torch.nn.Linear(hidden_dim2, hidden_dim2, bias=True),
            # torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim2, out_features, bias=True)
        )

    def forward(self, x):
        embedding = self.dnn1(x)
        preds = self.dnn2(torch.cat([x, embedding], axis=-1))

        return preds


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
def draw_loss(data_train, data_val, label="Loss"):
    """Plots the training and validation loss"""

    fig, ax = plt.subplots()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.plot(data_train, color='red', label='Training loss')
    ax.plot(data_val, color='blue', label='Validation loss')
    ax.legend(loc='upper right', borderpad=0.1, markerscale=0.1, fontsize='small')
    plt.savefig('loss-mlp.png')
    # plt.show()


# Let's draw the predictions
def draw_test(y_test, y_pred, bins, range, label1, label2, fig_name):
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