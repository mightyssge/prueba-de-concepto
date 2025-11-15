from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    mask = mask.to(dtype=torch.bool)
    min_val = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(~mask, min_val)
    return torch.softmax(scores, dim=dim)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.d_k = out_dim // heads
        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, N, F], adj: [B, N, N]
        B, N, _ = x.size()
        q = self.q_proj(x).view(B, N, self.heads, self.d_k)
        k = self.k_proj(x).view(B, N, self.heads, self.d_k)
        v = self.v_proj(x).view(B, N, self.heads, self.d_k)

        scores = torch.einsum("bnhd,bmhd->bhnm", q, k) / math.sqrt(self.d_k + 1e-9)
        # ensure at least self-connections
        if adj is None:
            adj_mask = torch.ones(B, N, N, device=x.device, dtype=torch.bool)
        else:
            adj_mask = adj.bool()
            eye = torch.eye(N, device=x.device, dtype=torch.bool).unsqueeze(0).expand(B, N, N)
            adj_mask = adj_mask | eye

        attn = masked_softmax(scores, adj_mask.unsqueeze(1), dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bhnm,bmhd->bnhd", attn, v)
        out = out.reshape(B, N, self.out_dim)
        out = self.out_proj(out)
        return F.relu(out)


class GraphEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(num_layers):
            layers.append(GraphAttentionLayer(dim, hidden_dim, heads=heads, dropout=dropout))
            dim = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        for gat in self.layers:
            x = gat(x, adj)
        return self.norm(x)


class RelationalAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.d_k = dim // heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, pooled: torch.Tensor, nodes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # pooled: [B, D], nodes: [B, N, D]
        B, N, _ = nodes.size()
        q = self.q_proj(pooled).view(B, self.heads, self.d_k)
        k = self.k_proj(nodes).view(B, N, self.heads, self.d_k)
        v = self.v_proj(nodes).view(B, N, self.heads, self.d_k)

        scores = torch.einsum("bhd,bnhd->bhn", q, k) / math.sqrt(self.d_k + 1e-9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bhn,bnhd->bhd", attn, v).reshape(B, self.dim)
        out = self.out(out)
        out = F.relu(out)
        return out, attn


class GraphActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        node_feat_dim: int,
        hidden_dim: int = 128,
        n_actions: int = 11,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.graph = GraphEncoder(node_feat_dim, hidden_dim, num_layers=2, heads=heads, dropout=dropout)
        self.state_enc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.rel_attn = RelationalAttention(hidden_dim, heads=heads, dropout=dropout)
        self.gru = nn.GRU(input_size=hidden_dim * 2, hidden_size=hidden_dim, batch_first=True)
        self.actor_head = nn.Linear(hidden_dim, n_actions)

    def forward(
        self,
        obs_vec: torch.Tensor,
        node_feats: torch.Tensor,
        adj: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # obs_vec: [B, F], node_feats: [B, N, Fn], adj: [B, N, N]
        graph_emb = self.graph(node_feats, adj)
        graph_pool = graph_emb.mean(dim=1)
        state_emb = self.state_enc(obs_vec)
        rel_out, attn = self.rel_attn(graph_pool, graph_emb)
        fused = torch.cat([rel_out, state_emb], dim=-1).unsqueeze(1)  # [B, 1, 2H]
        gru_out, next_hidden = self.gru(fused, hidden)
        logits = self.actor_head(gru_out.squeeze(1))
        return logits, next_hidden, attn


class CentralCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_vec: torch.Tensor) -> torch.Tensor:
        return self.net(state_vec).squeeze(-1)
