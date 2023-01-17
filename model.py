import torch
import torch.nn.functional as F
from torch.nn import Linear

class vertexTag(torch.nn.Module):
    def __init__(self, in_features: int=26, out_features: int=5, hidden_dim: int=64):
        super(vertexTag, self).__init__()
        self.fc1 = torch.nn.Linear(in_features, hidden_dim, bias=True)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc4 = torch.nn.Linear(hidden_dim, out_features, bias=True)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.softmax(self.fc4(x), dim=-1)
        x = F.relu(self.fc4(x))
        return x
    
    
class vertexReg(torch.nn.Module):
    def __init__(self, in_features: int=31, out_features: int=7, hidden_dim: int=32):
        super(vertexReg, self).__init__()
        self.fc1 = torch.nn.Linear(in_features, hidden_dim, bias=True)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc4 = torch.nn.Linear(hidden_dim, out_features, bias=True)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    

class AlexTag(torch.nn.Module):
    def __init__(self, num_classes: int=5, in_channels: int=26, out_channels: int=10):
        super(AlexTag, self).__init__()
        self.fc1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=2),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size = 3, stride = 2)
        )
        
        self.fc2 = torch.nn.Sequential(
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=2),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size = 3, stride = 2)
        )
        
        self.fc3 = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(out_channels*3*3*2*2, out_channels, bias=True),
            torch.nn.ReLU()
        )
        
        self.fc4 = torch.nn.Sequential(
            torch.nn.Linear(out_channels*3*3*2*2, out_channels, bias=True)
        )
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        
        return x
    


class vertexMLP(torch.nn.Module):
    def __init__(self, in_features: int=13, hotid_dim: int=5, vert_dim: int=6, hidden_dim1: int=64, hidden_dim2: int=64):
        super(vertexMLP, self).__init__()
        self.fc1 = vertexTag(in_features, hotid_dim, hidden_dim1)
        # self.fc1 = ConvTag()
        self.fc2 = vertexReg(in_features+hotid_dim, vert_dim, hidden_dim2)
        
    def forward(self, x):
        pred_hotid = self.fc1(x)
        y = torch.argmax(pred_hotid, dim=-1).squeeze()
        y = F.one_hot(y, num_classes=5).float()
        pred_vert = self.fc2(torch.cat([x, y], axis=-1))
        return pred_hotid, pred_vert
    

class dimuNet(torch.nn.Module):
    def __init__(self, in_features: int=26, num_classes: int=5, out_features: int=7, hidden_dim=50):
        super(dimuNet, self).__init__()
        self.tagger = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_dim, bias=True),
            # torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 30, bias=True),
            # torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 15, bias=True),
            # torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(15, num_classes, bias=True),
            torch.nn.ReLU()
            # torch.nn.Softmax(dim=-1)
        )
        
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(in_features+num_classes, hidden_dim, bias=True),
            # torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            # torch.nn.Dropout(0.4),
            torch.nn.Linear(hidden_dim, 30, bias=True),
            torch.nn.ReLU(),
            # torch.nn.Dropout(0.1),
            torch.nn.Linear(30, 15, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(15, out_features, bias=True)
        )
        
    
    def forward(self, x):
        pred_hotid = self.tagger(x)
        y = torch.argmax(pred_hotid, dim=-1).squeeze()
        y = F.one_hot(y, num_classes=5).float()
        pred_dim = self.regressor(torch.cat([x, y], axis=-1))
        return pred_hotid, pred_dim