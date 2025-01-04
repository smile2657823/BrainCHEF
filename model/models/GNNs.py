# 导入包
import torch
import torch.nn as nn
from .convs.graphs import GCNConv,GATConv,GINConv,GraphSAGEConv
import warnings
warnings.filterwarnings('ignore')


class GCN(torch.nn.Module):
    def __init__(self,input_channels,hidden_channels,num_layers):
        super(GCN, self).__init__()
        torch.manual_seed(403)
        self.relu = nn.ReLU(inplace=True)
        self.gnn = nn.ModuleList()
        for i in range(num_layers):
            if i==0:
                self.gnn.append(GCNConv(input_channels, hidden_channels))
            elif i==num_layers-1:
                self.gnn.append(GCNConv(hidden_channels, hidden_channels))
            else:
                self.gnn.append(GCNConv(hidden_channels, hidden_channels))
    def forward(self, x, g):
        for layer,(G,) in enumerate(zip(self.gnn)):
            x = G(x,g)
            x = self.relu(x)
        return x

class GraphSAGE(torch.nn.Module):
    def __init__(self,input_channels,hidden_channels,num_layers):
        super(GraphSAGE, self).__init__()
        torch.manual_seed(403)
        self.gnn = nn.ModuleList()
        for i in range(num_layers):
            if i==0:
                self.gnn.append(GraphSAGEConv(input_channels, hidden_channels))
            elif i==num_layers-1:
                self.gnn.append(GraphSAGEConv(hidden_channels, hidden_channels))
            else:
                self.gnn.append(GraphSAGEConv(hidden_channels, hidden_channels))
    def forward(self, x, g):
        for layer,(G,) in enumerate(zip(self.gnn)):
            x = G(x,g)
            x = x.relu()
        return x
    
class GAT(torch.nn.Module):
    def __init__(self,input_channels,hidden_channels,num_layers):
        super(GAT, self).__init__()
        torch.manual_seed(403)
        self.gnn = nn.ModuleList()
        for i in range(num_layers):
            if i==0:
                self.gnn.append(GATConv(input_channels, hidden_channels))
            elif i==num_layers-1:
                self.gnn.append(GATConv(hidden_channels, hidden_channels))
            else:
                self.gnn.append(GATConv(hidden_channels, hidden_channels))
    def forward(self, x, g):
        for layer,(G,) in enumerate(zip(self.gnn)):
            x = G(x,g)
            x = x.relu()
        return x
    
class GIN(torch.nn.Module):
    def __init__(self,input_channels,hidden_channels,num_layers):
        super(GIN, self).__init__()
        torch.manual_seed(403)
        self.gnn = nn.ModuleList()
        for i in range(num_layers):
            if i==0:
                self.gnn.append(GINConv(nn.Linear(input_channels, hidden_channels)))
            elif i==num_layers-1:
                self.gnn.append(GINConv(nn.Linear(input_channels, hidden_channels)))
            else:
                self.gnn.append(GINConv(nn.Linear(input_channels, hidden_channels)))
    def forward(self, x, g):
        for layer,(G,) in enumerate(zip(self.gnn)):
            x = G(x,g)
            x = x.relu()
        return x