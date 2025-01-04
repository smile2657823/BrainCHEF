import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv,GATConv,summary
from torch.nn import Linear
import numpy as np
from torch.nn import Sequential
# GCN
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv,GATConv,summary
from torch.nn import Linear
import numpy as np
from torch.nn import Sequential
# GCN
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=False, add_self_loops=False)
        self.conv2 = GCNConv(in_channels, hidden_channels, cached=False, add_self_loops=False)
        self.dropout = dropout
    def forward(self, x, edge_index,edge_weight):
        x = self.conv1(x, edge_index,edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index,edge_weight)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x
# GAT
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GAT, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, cached=False, add_self_loops=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels, hidden_channels, cached=False, add_self_loops=False))
        self.convs.append(GATConv(hidden_channels, out_channels, cached=True, add_self_loops=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index,edge_weight):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index,edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index,edge_weight)
        return x
# SAGE
class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, cached=False, add_self_loops=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels, cached=False, add_self_loops=False))
        self.convs.append(SAGEConv(hidden_channels, out_channels, cached=True, add_self_loops=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


import torch
import torch.nn as nn
import warnings
import torch.nn.functional as F
from .convs.hypergraphs.hypergraph import Hypergraph 
warnings.filterwarnings('ignore')

class CrossLevel(nn.Module):
    def __init__(self, in_channels, bias=True):
        super(CrossLevel, self).__init__()  # 确保先调用基类的__init__方法
        # 现在可以安全地添加子模块
        # print(in_channels*2)
        self.AR = nn.Sequential(
            nn.Linear(in_channels*2, 1, bias=bias),
            nn.Sigmoid()
        )
    def forward(self,Line_output,output,H_edge_index,H):
        AR_pairs = torch.cat((output[H_edge_index[0], :], Line_output[H_edge_index[1], :]), dim=1)
        # print(AR_pairs.shape)
        AR_coff = self.AR(AR_pairs).squeeze()
        A = torch.zeros((H.shape[0], H.shape[1]),device=AR_coff.device)
        Line_output = torch.matmul(A,Line_output)
        return Line_output
    

