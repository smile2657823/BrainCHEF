import torch.nn as nn
import torch
from einops import rearrange
import numpy as np
import math
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
class MeanReadout(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x, node_axis=1):
        return x.mean(node_axis)
class SERO(nn.Module):
    def __init__(self, hid_dim, dropout=0.2, upscale=1.0):
        super(SERO,self).__init__()
        self.q = nn.Linear(hid_dim, round(upscale*hid_dim))
        self.k = nn.Linear(hid_dim, round(upscale*hid_dim))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # print(x.shape)
        Q = self.q(x)
        K = self.k(x)
        attention = torch.sigmoid(torch.matmul(Q, rearrange(K, 'b n c -> b c n'))/np.sqrt(x.shape[-1]))
        attention = F.softmax(attention,dim=-1)
        output = torch.matmul(self.dropout(attention),x)
        output = output.mean(1)
        return output,attention
class MlpAttention(nn.Module):
    def __init__(self,input_dim,hidden_dim, dropout=0.5, upscale=1.0, **kwargs):
        super().__init__()
        self.embed_query = nn.Linear(hidden_dim, round(upscale*hidden_dim))
        self.embed_key = nn.Linear(hidden_dim, round(upscale*hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim*hidden_dim, hidden_dim)
        )
    def forward(self, x):
        x_q = self.embed_query(x)
        x_k = self.embed_key(x)
        attention = torch.sigmoid(torch.matmul(x_q, rearrange(x_k, 'b n c -> b c n'))/np.sqrt(x_q.shape[-1]))
        batch_size,nodes_num,hidden_dim = x.shape[:3]
        attention = F.softmax(attention,dim=-1)
        output = torch.matmul(self.dropout(attention),x)
        batch_size,nodes_num,hidden_dim = x.shape[:3]
        output = output.reshape(batch_size,nodes_num*hidden_dim)
        # print(output.shape)
        output = self.mlp(output)
        return output,attention