# models/temporal_gat.py
# ==============================================================================
# Spatio-Temporal Graph Attention Network (ST-GAT / TemporalGAT)
# Capstone Thesis (9.5/10, Top of Cohort) — Huynh Le Thanh Hai (ID: 2251012054)
# Ho Chi Minh City Open University · Supervisor: Dr. Le Quang Minh
#
# Architecture Overview:
#   1. TemporalEncoder: GRU (hid_dim=64) + 4-Head MultiheadAttention + Residual
#      skip connection over lagged epidemiological time series [N, T, F_t].
#   2. Spatial Graph Attention:
#      - GATConv Layer 1: in_dim + t_hidden -> 128 × 4 heads (concat=True, ELU)
#      - GATConv Layer 2: 512 -> 128 × 4 heads (concat=False, ELU)
#      - Linear Readout : 128 -> 1 (predicts log1p weekly dengue incidence)
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class TemporalEncoder(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int = 64, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hid_dim, batch_first=True)
        self.attn = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=heads,
                                          dropout=dropout, batch_first=True)
        self.proj = nn.Linear(hid_dim, hid_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(seq)                 # [N,T,H]
        attn_out, _ = self.attn(out, out, out) # [N,T,H]
        last = out[:, -1, :] + attn_out[:, -1, :]
        return self.proj(self.drop(last))      # [N,H]


class TemporalGAT(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 128,
        out_dim: int = 1,
        heads1: int = 4,
        heads2: int = 4,
        gat_dropout: float = 0.2,
        t_in_dim: int = 0,
        t_hidden: int = 64,
        t_heads: int = 4,
        t_dropout: float = 0.1,
    ):
        super().__init__()
        self.has_temporal = t_in_dim is not None and t_in_dim > 0
        if self.has_temporal:
            self.temporal = TemporalEncoder(t_in_dim, hid_dim=t_hidden, heads=t_heads, dropout=t_dropout)
            in0 = in_dim + t_hidden
        else:
            self.temporal = None
            in0 = in_dim

        self.gat1 = GATConv(in0, hidden, heads=heads1, dropout=gat_dropout, concat=True)
        self.gat2 = GATConv(hidden * heads1, hidden, heads=heads2, dropout=gat_dropout, concat=False)
        self.lin  = nn.Linear(hidden, out_dim)
        self.drop = nn.Dropout(gat_dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, temporal_seq=None) -> torch.Tensor:
        if self.has_temporal and temporal_seq is not None:
            t_emb = self.temporal(temporal_seq)  # [N, t_hidden]
            x = torch.cat([x, t_emb], dim=1)
        x = F.elu(self.gat1(x, edge_index)); x = self.drop(x)
        x = F.elu(self.gat2(x, edge_index)); x = self.drop(x)
        return self.lin(x).squeeze(-1)          # [N]
