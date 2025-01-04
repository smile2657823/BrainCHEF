from .network import GAT, GCN, SAGE
import torch
import torch.nn as nn
import warnings
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from functools import partial
import warnings
warnings.filterwarnings('ignore')
def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def sig_loss(x, y):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (x * y).sum(1)
    loss = torch.sigmoid(-loss)
    loss = loss.mean()
    return loss

def ModelChoose(model_type,
                in_channels,
                hidden_channels,
                out_channels,
                num_layers,
                dropout):
    if model_type == "GCN":
        model = GCN(in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    out_channels=out_channels,
                    num_layers=num_layers,
                    dropout=dropout)
    if model_type == "GAT":
        model = GAT(in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    out_channels=out_channels,
                    num_layers=num_layers,
                    dropout=dropout)
    if model_type == "SAGE":
        model = SAGE(in_channels=in_channels,
                     hidden_channels=hidden_channels,
                     out_channels=out_channels,
                     num_layers=num_layers,
                     dropout=dropout)
    if model_type == "MLP":
        model = nn.Sequential(nn.Linear(in_channels, hidden_channels * 2),
                              nn.PReLU(), nn.Dropout(0.2),
                              nn.Linear(hidden_channels * 2, out_channels))
    return model

def setup_loss_fn(loss_fn="sce", alpha=2):
    if loss_fn == "mse":
        criterion = nn.MSELoss()
    else:
        criterion = partial(sce_loss, alpha=alpha)
    return criterion

class PreModel(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 encoder_type="GCN",
                 decoder_type="GCN",
                 loss_fn="sce",
                 alpha=2,
                 replace_rate=0.1,
                 mask_rate=0.3,
                 remask=False):
        super(PreModel, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.loss_fn = loss_fn
        self.alpha = alpha
        self.replace_rate = replace_rate
        self.mask_token_rate = 1 - self.replace_rate
        self.mask_rate = mask_rate
        self.remask = remask
        dec_in_dim = out_channels
        # 编码器
        self.encoder = ModelChoose(model_type=self.encoder_type,
                                   in_channels=self.in_channels,
                                   hidden_channels=self.hidden_channels,
                                   out_channels=self.out_channels,
                                   num_layers=self.num_layers,
                                   dropout=self.dropout)
        # 解码器
        self.decoder = ModelChoose(model_type=self.decoder_type,
                                   in_channels=self.in_channels,
                                   hidden_channels=self.hidden_channels,
                                   out_channels=self.out_channels,
                                   num_layers=self.num_layers,
                                   dropout=self.dropout)
        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.in_channels))
        self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)
        self.criterion = setup_loss_fn(self.loss_fn, self.alpha)
    def encoding_mask_noise(self, x):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(self.mask_rate * num_nodes)
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]
        if self.replace_rate > 0:
            num_mask_nodes = len(mask_nodes)
            num_noise_nodes = int(self.replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[:int(self.mask_token_rate *
                                                    num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self.replace_rate *
                                                    num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(
                num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0
        out_x[token_nodes] += self.enc_mask_token
        return out_x, mask_nodes, keep_nodes
    def forward(self,x,edge_index,edge_weight):
        enc_x, mask_nodes, keep_nodes = self.encoding_mask_noise(x)
        # print(enc_x)
        enc_x = self.encoder(enc_x, edge_index, edge_weight)
        dec_x = self.encoder_to_decoder(enc_x)
        if self.remask:
            dec_x[mask_nodes] = 0
            dec_x = self.decoder(dec_x, edge_index, edge_weight)
        else:
            dec_x = self.decoder(dec_x, edge_index, edge_weight)
        loss = self.criterion(dec_x[mask_nodes], x[mask_nodes])
        return enc_x, loss