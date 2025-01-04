# 导入包
import torch
import torch.nn as nn
from .convs.hypergraphs import UniGATConv,UniSAGEConv,UniGCNConv,UniGINConv
import warnings
warnings.filterwarnings('ignore')

class UniGCN(torch.nn.Module):
    def __init__(self,input_channels,hidden_channels,num_layers):
        super(UniGCN, self).__init__()
        torch.manual_seed(403)
        self.hgnn = nn.ModuleList()
        for i in range(num_layers):
            if i==0:
                self.hgnn.append(UniGCNConv(input_channels, hidden_channels))
            elif i==num_layers-1:
                self.hgnn.append(UniGCNConv(hidden_channels, hidden_channels))
            else:
                self.hgnn.append(UniGCNConv(hidden_channels, hidden_channels))
    def forward(self, x, hg):
        for layer,(G,) in enumerate(zip(self.hgnn)):
            x = G(x,hg)
            x = x.relu()
        return x
    
class UniGAT(torch.nn.Module):
    def __init__(self,input_channels,hidden_channels,num_layers):
        super(UniGAT, self).__init__()
        torch.manual_seed(403)
        self.hgnn = nn.ModuleList()
        for i in range(num_layers):
            if i==0:
                self.hgnn.append(UniGATConv(input_channels, hidden_channels))
            elif i==num_layers-1:
                self.hgnn.append(UniGATConv(hidden_channels, hidden_channels))
            else:
                self.hgnn.append(UniGATConv(hidden_channels, hidden_channels))
    def forward(self, x, hg):
        for layer,(G,) in enumerate(zip(self.hgnn)):
            x = G(x,hg)
            x = x.relu()
        return x
    
class UniSAGE(torch.nn.Module):
    def __init__(self,input_channels,hidden_channels,num_layers):
        super(UniSAGE, self).__init__()
        torch.manual_seed(403)
        self.hgnn = nn.ModuleList()
        for i in range(num_layers):
            if i==0:
                self.hgnn.append(UniSAGEConv(input_channels, hidden_channels))
            elif i==num_layers-1:
                self.hgnn.append(UniSAGEConv(hidden_channels, hidden_channels))
            else:
                self.hgnn.append(UniSAGEConv(hidden_channels, hidden_channels))
    def forward(self, x, hg):
        for layer,(G,) in enumerate(zip(self.hgnn)):
            x = G(x,hg)
            x = x.relu()
        return x
    
class UniGIN(torch.nn.Module):
    def __init__(self,input_channels,hidden_channels,num_layers):
        super(UniGIN, self).__init__()
        torch.manual_seed(403)
        self.hgnn = nn.ModuleList()
        for i in range(num_layers):
            if i==0:
                self.hgnn.append(UniGINConv(input_channels, hidden_channels))
            elif i==num_layers-1:
                self.hgnn.append(UniGINConv(hidden_channels, hidden_channels))
            else:
                self.hgnn.append(UniGINConv(hidden_channels, hidden_channels))
    def forward(self, x, hg):
        for layer,(G,) in enumerate(zip(self.hgnn)):
            x = G(x,hg)
            x = x.relu()
        return x