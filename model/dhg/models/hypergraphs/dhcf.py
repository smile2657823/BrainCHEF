from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dhg.structure.hypergraphs import Hypergraph


class DHCF(nn.Module):
    r"""The DHCF model proposed in `Dual Channel Hypergraph Collaborative Filtering <https://dl.acm.org/doi/10.1145/3394486.3403253>`_ paper (KDD 2020).
    
    .. note::

        The user and item embeddings and trainable parameters are initialized with xavier_uniform distribution.
    
    Args:
        ``num_users`` (``int``): The Number of users.
        ``num_items`` (``int``): The Number of items.
        ``emb_dim`` (``int``): Embedding dimension.
        ``num_layers`` (``int``): The Number of layers. Defaults to ``3``.
        ``drop_rate`` (``float``): The dropout probability. Defaults to ``0.5``.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        emb_dim: int,
        num_layers: int = 3,
        drop_rate: float = 0.5,
    ) -> None:

        super().__init__()
        self.num_users, self.num_items = num_users, num_items
        self.num_layers = num_layers
        self.drop_rate = drop_rate
        self.u_embedding = nn.Embedding(num_users, emb_dim)
        self.i_embedding = nn.Embedding(num_items, emb_dim)
        self.W_gc, self.W_bi = nn.ModuleList(), nn.ModuleList()
        for _ in range(self.num_layers):
            self.W_gc.append(nn.Linear(emb_dim, emb_dim))
            self.W_bi.append(nn.Linear(emb_dim, emb_dim))
        self.reset_parameters()

    def reset_parameters(self):
        r"""Initialize learnable parameters.
        """
        nn.init.xavier_uniform_(self.u_embedding.weight)
        nn.init.xavier_uniform_(self.i_embedding.weight)
        for W_gc, W_bi in zip(self.W_gc, self.W_bi):
            nn.init.xavier_uniform_(W_gc.weight)
            nn.init.xavier_uniform_(W_bi.weight)
            nn.init.constant_(W_gc.bias, 0)
            nn.init.constant_(W_bi.bias, 0)

    def forward(
        self, hg_ui: Hypergraph, hg_iu: Hypergraph
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""The forward function.

        Args:
            ``hg_ui`` (``dhg.Hypergraph``): The hypergraph structure that users as vertices.
            ``hg_iu`` (``dhg.Hypergraph``): The hypergraph structure that items as vertices.
        """
        u_embs = self.u_embedding.weight
        i_embs = self.i_embedding.weight
        all_embs = torch.cat([u_embs, i_embs], dim=0)

        embs_list = [all_embs]
        for _idx in range(self.num_layers):
            u_embs, i_embs = torch.split(
                all_embs, [self.num_users, self.num_items], dim=0
            )
            #==========================================================================================
            # Two JHConv Layers for users and items, respectively.
            u_embs = hg_ui.smoothing_with_HGNN(u_embs)
            i_embs = hg_iu.smoothing_with_HGNN(i_embs)
            g_embs = torch.cat([u_embs, i_embs], dim=0)
            sum_embs = F.leaky_relu(self.W_gc[_idx](g_embs) + all_embs, negative_slope=0.2)
            #==========================================================================================

            bi_embs = all_embs * g_embs
            bi_embs = F.leaky_relu(self.W_bi[_idx](bi_embs), negative_slope=0.2)

            all_embs = sum_embs + bi_embs
            all_embs = F.dropout(all_embs, p=self.drop_rate, training=self.training)
            all_embs = F.normalize(all_embs, p=2, dim=1)

            embs_list.append(all_embs)
        embs = torch.stack(embs_list, dim=1)
        embs = torch.mean(embs, dim=1)

        u_embs, i_embs = torch.split(embs, [self.num_users, self.num_items], dim=0)
        return u_embs, i_embs
