# 导入包
import torch
import torch.nn as nn
from .convs.hypergraphs import HyperGCNConv,HGNNConv,HGNNPConv,HNHNConv,JHConv
import warnings
import torch.nn.functional as F
from .convs.hypergraphs.hypergraph import Hypergraph 
warnings.filterwarnings('ignore')


class HGNN(torch.nn.Module):
    def __init__(self,input_channels,hidden_channels,num_layers,dropout):
        super(HGNN, self).__init__()
        torch.manual_seed(403)
        self.hgnn = nn.ModuleList()
        for i in range(num_layers):
            if i==0:
                self.hgnn.append(HGNNConv(input_channels, hidden_channels))
            elif i==num_layers-1:
                self.hgnn.append(HGNNConv(hidden_channels, hidden_channels))
            else:
                self.hgnn.append(HGNNConv(hidden_channels, hidden_channels))
        self.dropout = dropout
    def forward(self, x, hg):
        for layer,(G,) in enumerate(zip(self.hgnn)):
            x = G(x,hg)
            x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class HGNNP(torch.nn.Module):
    def __init__(self,input_channels,hidden_channels,num_layers,dropout):
        super(HGNNP, self).__init__()
        torch.manual_seed(403)
        self.hgnn = nn.ModuleList()
        for i in range(num_layers):
            if i==0:
                self.hgnn.append(HGNNPConv(input_channels, hidden_channels))
            elif i==num_layers-1:
                self.hgnn.append(HGNNPConv(hidden_channels, hidden_channels))
            else:
                self.hgnn.append(HGNNPConv(hidden_channels, hidden_channels))
        self.dropout = dropout
    def forward(self, x, hg):
        for layer,(G,) in enumerate(zip(self.hgnn)):
            x = G(x,hg)
            x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class JHGNN(torch.nn.Module):
    def __init__(self,input_channels,hidden_channels,num_layers,dropout):
        super(JHGNN, self).__init__()
        torch.manual_seed(403)
        self.hgnn = nn.ModuleList()
        for i in range(num_layers):
            if i==0:
                self.hgnn.append(JHConv(input_channels, hidden_channels))
            elif i==num_layers-1:
                self.hgnn.append(JHConv(hidden_channels, hidden_channels))
            else:
                self.hgnn.append(JHConv(hidden_channels, hidden_channels))
        self.dropout = dropout
    def forward(self, x, hg):
        for layer,(G,) in enumerate(zip(self.hgnn)):
            x = G(x,hg)
            x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
class HyperGCN(torch.nn.Module):
    def __init__(self,input_channels,hidden_channels,num_layers,dropout):
        super(HyperGCN, self).__init__()
        torch.manual_seed(403)
        self.hgnn = nn.ModuleList()
        for i in range(num_layers):
            if i==0:
                self.hgnn.append(HyperGCNConv(input_channels, hidden_channels))
            elif i==num_layers-1:
                self.hgnn.append(HyperGCNConv(hidden_channels, hidden_channels))
            else:
                self.hgnn.append(HyperGCNConv(hidden_channels, hidden_channels))
        self.dropout = dropout
    def forward(self, x, hg):
        for layer,(G,) in enumerate(zip(self.hgnn)):
            x = G(x,hg)
            x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
class HNHN(torch.nn.Module):
    def __init__(self,input_channels,hidden_channels,num_layers):
        super(HNHN, self).__init__()
        torch.manual_seed(403)
        self.hgnn = nn.ModuleList()
        for i in range(num_layers):
            if i==0:
                self.hgnn.append(HNHNConv(input_channels, hidden_channels))
            elif i==num_layers-1:
                self.hgnn.append(HNHNConv(hidden_channels, hidden_channels))
            else:
                self.hgnn.append(HNHNConv(hidden_channels, hidden_channels))
    def forward(self, x, hg):
        for layer,(G,) in enumerate(zip(self.hgnn)):
            x = G(x,hg)
            x = x.relu()
        return x

class HANConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        use_bn: bool = True,
        drop_rate: float = 0.2,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.W1 = nn.Linear(in_channels, out_channels, bias=bias)
        self.W2 = nn.Linear(in_channels, out_channels, bias=bias)
        self.MLP_node = nn.Sequential(nn.Linear(in_channels*2, 1, bias=bias),nn.Sigmoid())
        self.MLP_edge = nn.Sequential(nn.Linear(in_channels*2, 1, bias=bias),nn.Sigmoid())

    def forward(self, x: torch.Tensor, hg: Hypergraph,H_edge_index,H_edge_weight,A_edge_index,A_edge_weight,H,A,X_L,aggr = "softmax_then_sum"):
        assert aggr in ["mean", "sum", "softmax_then_sum"]
        x_n = self.W1(x)
        # x_n = x
        # for index in  range(A_edge_index.shape[1]):
        #     A_edge_weight[index] = self.MLP_node(torch.cat([x_n[A_edge_index[0][index],:],x_n[A_edge_index[1][index],:]],dim=0))
        # X_L = X_L.T
        # print(X_L.shape)
        # print(x_n.shape)
        X_L = self.W2(X_L)
        node_pairs = torch.cat((x_n[A_edge_index[0], :], x_n[A_edge_index[1], :]), dim=1)
        A_node = self.MLP_node(node_pairs).squeeze()
        A = torch.zeros((A.shape[0], A.shape[1]), device=A_node.device)
        A[A_edge_index[0],A_edge_index[1]] = A_node
        A_node = torch.matmul(A, H).t()
        # A_node = nn.Sigmoid(A_node)
        # A_node = torch.sparse_coo_tensor(A_edge_index, A_edge_weight, H.shape).coalesce().to_dense() 
        # sparse_H = torch.sparse_coo_tensor(H_edge_index, H_edge_weight, A.shape).coalesce()
        # A_node = torch.sparse.mm(sparse_H, sparse_A).t().to_dense()
        # A_node = F.softmax(A_node, dim=1)
        A_node = A_node[H_edge_index[1], H_edge_index[0]]
        # A_node = H_edge_weight
        # for index in  range(H_edge_index.shape[1]):
        #     H_edge_weight[index] = A_node[H_edge_index[1][index],H_edge_index[0][index]]
        X_L = hg.v2e_aggregation( X = x_n,aggr = aggr, v2e_weight = A_node)
        # x_e = self.W2(x_e)
        # for index in  range(H_edge_index.shape[1]):
        #     H_edge_weight[index] = self.MLP_edge(torch.cat([x_n[H_edge_index[0][index],:],x_e[H_edge_index[1][index],:]],dim=0))
        edge_pairs = torch.cat((x_n[H_edge_index[0], :], X_L[H_edge_index[1], :]), dim=1)
        A_edge = self.MLP_edge(edge_pairs).squeeze()
        # A_edge = nn.Sigmoid(A_edge)
        # A_edge = H_edge_weight
        # A_node = F.softmax(A_node, dim=1)
        x_n = hg.e2v_aggregation( X = X_L, aggr = aggr,e2v_weight = A_edge)
        x_n = x_n + x
        # x_e = x_e + X_L
        if not self.is_last:
            x_n = self.act(x_n)
            if self.bn is not None:
                x_n = self.bn(x_n)
            x_n = self.drop(x_n)
        return x_n,X_L

class HAN(torch.nn.Module):
    def __init__(self,input_channels,hidden_channels,num_layers,dropout):
        super(HAN, self).__init__()
        torch.manual_seed(403)
        self.hgnn = nn.ModuleList()
        for i in range(num_layers):
            if i==0:
                self.hgnn.append(HANConv(input_channels, hidden_channels))
            elif i==num_layers-1:
                self.hgnn.append(HANConv(hidden_channels, hidden_channels))
            else:
                self.hgnn.append(HANConv(hidden_channels, hidden_channels))
        self.dropout = dropout
    def forward(self, x, hg,H_edge_index,H_edge_weight,A_edge_index,A_edge_weight,H,A,X_L,aggr = "mean"):
        for layer,(G,) in enumerate(zip(self.hgnn)):
            x,X_L = G(x,hg,H_edge_index,H_edge_weight,A_edge_index,A_edge_weight,H,A,X_L,aggr = aggr)
            x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x