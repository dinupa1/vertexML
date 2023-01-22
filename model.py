import torch
import torch.nn.functional as F
from torch.nn import Linear

class Tagger(torch.nn.Module):
    def __init__(self, in_features=26, out_features=5):
        super(Tagger, self).__init__()
        self.fc1 = torch.nn.Linear(in_features, 124, bias=True)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        self.fc2 = torch.nn.Linear(124, 124, bias=True)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        self.fc3 = torch.nn.Linear(124, 124, bias=True)
        torch.nn.init.xavier_normal_(self.fc3.weight)
        # self.dropout = torch.nn.Dropout(0.1)
        self.fc4 = torch.nn.Linear(124, out_features, bias=True)
        torch.nn.init.xavier_normal_(self.fc4.weight)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = self.dropout(x)
        x = F.relu(self.fc4(x))
        return x
    
    
class Regressor(torch.nn.Module):
    def __init__(self, in_features=31, out_features=7):
        super(Regressor, self).__init__()
        self.fc1 = torch.nn.Linear(in_features, 248, bias=True)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        # self.norm1 = torch.nn.BatchNorm1d(50)
        self.fc2 = torch.nn.Linear(248, 248, bias=True)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        self.fc3 = torch.nn.Linear(248, 248, bias=True)
        torch.nn.init.xavier_normal_(self.fc3.weight)
        self.fc4 = torch.nn.Linear(248, 248, bias=True)
        torch.nn.init.xavier_normal_(self.fc4.weight)
        self.fc5 = torch.nn.Linear(248, out_features, bias=True)
        torch.nn.init.xavier_normal_(self.fc5.weight)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = self.norm1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    

    
class ConvTagger(torch.nn.Module):
    def __init__(self):
        super(ConvTagger, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 10, 5, padding=2)
        self.pool1 = torch.nn.AvgPool1d(2)
        self.conv2 = torch.nn.Conv2d(10, 20, 5)
        self.pool2 = torch.nn.AvgPool1d(2)
        self.fc1 = torch.nn.Linear(20* 10, 120)
        self.fc2 = torch.nn.Linear(120, 5)

    def forward(self, x):
        x = x.unsqueeze(0)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
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