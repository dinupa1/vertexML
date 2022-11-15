import torch
import torch.nn.functional as F
from torch.nn import Linear

class vertexTag(torch.nn.Module):
    def __init__(self, in_features: int=13, out_features: int=5, hidden_dim: int=64):
        super(vertexTag, self).__init__()
        self.fc1 = torch.nn.Linear(in_features, hidden_dim, bias=True)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
        # self.fc4 = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc5 = torch.nn.Linear(hidden_dim, out_features, bias=True)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.log_softmax(self.fc4(x), dim=1)
        x = self.fc5(x)
        return x
    
    
class vertexReg(torch.nn.Module):
    def __init__(self, in_features: int=13, out_features: int=6, hidden_dim: int=32):
        super(vertexReg, self).__init__()
        self.fc1 = torch.nn.Linear(in_features, hidden_dim, bias=True)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc4 = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc5 = torch.nn.Linear(hidden_dim, out_features, bias=True)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        x = self.fc5(x)
        return x
    


class vertexMLP(torch.nn.Module):
    def __init__(self, in_features: int=13, hotid_dim: int=5, vert_dim: int=6, hidden_dim1: int=64, hidden_dim2: int=64):
        super(vertexMLP, self).__init__()
        self.fc1 = vertexTag(in_features, hotid_dim, hidden_dim1)
        self.fc2 = vertexReg(in_features+hotid_dim, vert_dim, hidden_dim2)
        
    def forward(self, x):
        pred_hotid = self.fc1(x)
        pred_vert = self.fc2(torch.cat([x, pred_hotid], axis=-1))
        return pred_hotid, pred_vert