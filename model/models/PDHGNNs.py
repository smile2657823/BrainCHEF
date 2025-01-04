# 导入包
import torch
import torch.nn as nn
from .convs.hypergraphs import HyperGCNConv,HGNNConv,HGNNPConv,HNHNConv,JHConv
import torch.nn.functional as F
from .pointnet_utils import TopologicalFeatureNetwork
import warnings
warnings.filterwarnings('ignore')


class PDHGNN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers,dropout=0.0):
        super(PDHGNN, self).__init__()
        torch.manual_seed(403)
        self.topology_branch = TopologicalFeatureNetwork(channels=5, hidden_size = input_channels)
        self.hgnn = nn.ModuleList()
        self.topology_linear = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        for i in range(num_layers):
            if i == 0:
                self.hgnn.append(HGNNConv(input_channels, hidden_channels))
                self.topology_linear.append(nn.Linear(input_channels,hidden_channels))
            elif i == num_layers - 1:
                self.hgnn.append(HGNNConv(hidden_channels, hidden_channels))
                self.topology_linear.append(nn.Linear(hidden_channels,hidden_channels))
            else:
                self.hgnn.append(HGNNConv(hidden_channels, hidden_channels))
                self.topology_linear.append(nn.Linear(hidden_channels,hidden_channels))

    def forward(self, x, hg,pd):
        topo = self.topology_branch(pd)
        topo1 = topo
        for layer, (G,T) in enumerate(zip(self.hgnn,self.topology_linear)):
            x = G(x, hg)
            topo = T(topo)
            x = x + x*self.dropout(topo)
            x = x.relu()
        return x,topo1

class PDHGNNP(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers,dropout=0.2):
        super(PDHGNNP, self).__init__()
        torch.manual_seed(403)
        self.topology_branch = TopologicalFeatureNetwork(channels=5, hidden_size = input_channels)
        self.hgnn = nn.ModuleList()
        self.topology_linear = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        for i in range(num_layers):
            if i == 0:
                self.hgnn.append(HGNNPConv(input_channels, hidden_channels))
                self.topology_linear.append(nn.Linear(input_channels,hidden_channels))
            elif i == num_layers - 1:
                self.hgnn.append(HGNNPConv(hidden_channels, hidden_channels))
                self.topology_linear.append(nn.Linear(hidden_channels,hidden_channels))
            else:
                self.hgnn.append(HGNNPConv(hidden_channels, hidden_channels))
                self.topology_linear.append(nn.Linear(hidden_channels,hidden_channels))

    def forward(self, x, hg,pd):
        topo = self.topology_branch(pd)
        for layer, (G,T) in enumerate(zip(self.hgnn,self.topology_linear)):
            x = G(x, hg)
            topo = T(topo)
            x = x + x*self.dropout(topo)
            x = x.relu()
        return x,topo

    
class PDJHGNN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers,dropout=0.2):
        super(PDJHGNN, self).__init__()
        torch.manual_seed(403)
        self.topology_branch = TopologicalFeatureNetwork(channels=5, hidden_size = input_channels)
        self.hgnn = nn.ModuleList()
        self.topology_linear = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        for i in range(num_layers):
            if i == 0:
                self.hgnn.append(JHConv(input_channels, hidden_channels))
                self.topology_linear.append(nn.Linear(input_channels,hidden_channels))
            elif i == num_layers - 1:
                self.hgnn.append(JHConv(hidden_channels, hidden_channels))
                self.topology_linear.append(nn.Linear(hidden_channels,hidden_channels))
            else:
                self.hgnn.append(JHConv(hidden_channels, hidden_channels))
                self.topology_linear.append(nn.Linear(hidden_channels,hidden_channels))

    def forward(self, x, hg,pd):
        topo = self.topology_branch(pd)
        for layer, (G,T) in enumerate(zip(self.hgnn,self.topology_linear)):
            x = G(x, hg)
            topo = T(topo)
            x = x + x*self.dropout(topo)
            x = x.relu()
        return x,topo
    
class PDHNHN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers,dropout=0.2):
        super(PDHNHN, self).__init__()
        torch.manual_seed(403)
        self.topology_branch = TopologicalFeatureNetwork(channels=5, hidden_size = input_channels)
        self.hgnn = nn.ModuleList()
        self.topology_linear = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        for i in range(num_layers):
            if i == 0:
                self.hgnn.append(HNHNConv(input_channels, hidden_channels))
                self.topology_linear.append(nn.Linear(input_channels,hidden_channels))
            elif i == num_layers - 1:
                self.hgnn.append(HNHNConv(hidden_channels, hidden_channels))
                self.topology_linear.append(nn.Linear(hidden_channels,hidden_channels))
            else:
                self.hgnn.append(HNHNConv(hidden_channels, hidden_channels))
                self.topology_linear.append(nn.Linear(hidden_channels,hidden_channels))

    def forward(self, x, hg,pd):
        topo = self.topology_branch(pd)
        for layer, (G,T) in enumerate(zip(self.hgnn,self.topology_linear)):
            x = G(x, hg)
            topo = T(topo)
            x = x + x*self.dropout(topo)
            x = x.relu()
        return x,topo

class PDHyperGCN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers,dropout=0.2):
        super(PDHyperGCN, self).__init__()
        torch.manual_seed(403)
        self.topology_branch = TopologicalFeatureNetwork(channels=5, hidden_size = input_channels)
        self.hgnn = nn.ModuleList()
        self.topology_linear = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        # print(num_layers)
        for i in range(num_layers):
            if i == 0:
                self.hgnn.append(HyperGCNConv(input_channels, hidden_channels))
                self.topology_linear.append(nn.Linear(input_channels,hidden_channels))
            elif i == num_layers - 1:
                self.hgnn.append(HyperGCNConv(hidden_channels, hidden_channels))
                self.topology_linear.append(nn.Linear(hidden_channels,hidden_channels))
            else:
                self.hgnn.append(HyperGCNConv(hidden_channels, hidden_channels))
                self.topology_linear.append(nn.Linear(hidden_channels,hidden_channels))

    def forward(self, x, hg,pd):
        topo = self.topology_branch(pd)
        for layer, (G,T) in enumerate(zip(self.hgnn,self.topology_linear)):
            x = G(x, hg)
            topo = T(topo)
            x = x + x*self.dropout(topo)
            x = x.relu()
        return x,topo
