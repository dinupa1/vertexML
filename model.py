import torch
import torch.nn.functional as F
from torch.nn import Linear

class Tagger(torch.nn.Module):
    def __init__(self, in_features=26, out_features=5):
        super(Tagger, self).__init__()
        self.fc1 = torch.nn.Linear(in_features, 50, bias=True)
        self.fc2 = torch.nn.Linear(50, 50, bias=True)
        self.fc3 = torch.nn.Linear(50, 30, bias=True)
        self.fc4 = torch.nn.Linear(30, out_features, bias=True)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x
    
    
class Regressor(torch.nn.Module):
    def __init__(self, in_features=31, out_features=7):
        super(Regressor, self).__init__()
        self.fc1 = torch.nn.Linear(in_features, 50, bias=True)
        self.fc2 = torch.nn.Linear(50, 50, bias=True)
        self.fc3 = torch.nn.Linear(50, 20, bias=True)
        self.fc4 = torch.nn.Linear(20, out_features, bias=True)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    

    
class LeNet5Tagger(torch.nn.Module):
    def __init__(self):
        super(LeNet5Tagger, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5, padding=2)
        self.pool1 = torch.nn.AvgPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.pool2 = torch.nn.AvgPool2d(2, 2)
        self.fc1 = torch.nn.Linear(16*4*4, 120)
        self.fc2 = torch.nn.Linear(120, 5)

    def forward(self, x):
        x = x.unsqueeze(0).unsqueeze(0)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)
    

class dimuNet(torch.nn.Module):
    def __init__(self):
        super(dimuNet, self).__init__()
        self.tagger = Tagger()
        self.regressor = Regressor()
        
    
    def forward(self, x):
        pred_hotid = self.tagger(x)
        out = torch.argmax(pred_hotid, dim=-1).squeeze()
        out = F.one_hot(out, num_classes=5).float()
        pred_vert = self.regressor(torch.cat([x, out], axis=-1))
        return pred_hotid, pred_vert
    


# class dimuNet(torch.nn.Module):
#     def __init__(self, in_features: int=26, num_classes: int=5, out_features: int=7):
#         super(dimuNet, self).__init__()
#         self.tagger = torch.nn.Sequential(
#             torch.nn.Linear(in_features, 50, bias=True),
#             # torch.nn.BatchNorm1d(hidden_dim),
#             torch.nn.ReLU(),
#             torch.nn.Linear(50, 50, bias=True),
#             # torch.nn.BatchNorm1d(hidden_dim),
#             torch.nn.ReLU(),
#             torch.nn.Linear(50, 20, bias=True),
#             # torch.nn.BatchNorm1d(hidden_dim),
#             torch.nn.ReLU(),
#             torch.nn.Linear(20, num_classes, bias=True),
#             torch.nn.ReLU()
#             # torch.nn.Softmax(dim=-1)
#         )
        
#         self.regressor = torch.nn.Sequential(
#             torch.nn.Linear(in_features+num_classes, 50, bias=True),
#             # torch.nn.BatchNorm1d(hidden_dim),
#             torch.nn.ReLU(),
#             # torch.nn.Dropout(0.4),
#             torch.nn.Linear(50, 50, bias=True),
#             torch.nn.ReLU(),
#             # torch.nn.Dropout(0.1),
#             torch.nn.Linear(50, 20, bias=True),
#             torch.nn.ReLU(),
#             torch.nn.Linear(20, out_features, bias=True)
#         )
        
    
#     def forward(self, x):
#         pred_hotid = self.tagger(x)
#         out = torch.argmax(pred_hotid, dim=-1).squeeze()
#         out = F.one_hot(out, num_classes=5).float()
#         pred_dim = self.regressor(torch.cat([x, out], axis=-1))
#         return pred_hotid, pred_dim