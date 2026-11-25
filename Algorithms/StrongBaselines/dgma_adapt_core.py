import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data
import os
import sys
_cur_dir = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.dirname(os.path.dirname(_cur_dir))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)
from Algorithms.Baselines.gat_ppo_encoder import encode_dag_gat_ppo_state, NODE_FEATURE_DIM

def _scatter_softmax(score: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if score.numel() == 0:
        return score
    score_max = torch.full((num_nodes,), float('-inf'), device=score.device, dtype=score.dtype)
    score_max.scatter_reduce_(0, index, score, reduce='amax', include_self=True)
    score_max = torch.where(torch.isfinite(score_max), score_max, torch.zeros_like(score_max))
    exp = torch.exp(score - score_max[index])
    sum_exp = torch.zeros(num_nodes, device=score.device, dtype=score.dtype)
    sum_exp.index_add_(0, index, exp)
    sum_exp = sum_exp.clamp_min(1e-12)
    return exp / sum_exp[index]

class FEDGLayer(nn.Module):

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.w1 = nn.Linear(hidden_dim, 1, bias=False)
        self.w2 = nn.Linear(hidden_dim, 1, bias=False)
        self.edge_bias = nn.Parameter(torch.zeros(1))
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N, H = h.shape
        if edge_index.numel() == 0:
            zero_msg = torch.zeros_like(h)
            return self.gru(zero_msg, h)
        src_orig, dst_orig = (edge_index[0], edge_index[1])
        src_rev, dst_rev = (dst_orig, src_orig)
        score = self.w1(h[dst_rev]).squeeze(-1) + self.w2(h[src_rev]).squeeze(-1) + self.edge_bias
        score = F.leaky_relu(score, 0.2)
        alpha = _scatter_softmax(score, dst_rev, num_nodes=N)
        msg = self.msg_proj(h[src_rev]) * alpha.unsqueeze(-1)
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst_rev, msg)
        h_new = self.gru(agg, h)
        return h_new

class FEDGEncoder(nn.Module):

    def __init__(self, input_dim: int=NODE_FEATURE_DIM, hidden_dim: int=64, num_layers: int=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([FEDGLayer(hidden_dim) for _ in range(num_layers)])
        self.act = nn.ReLU()
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.act(self.input_proj(x))
        for layer in self.layers:
            h = layer(h, edge_index)
        return h

def _end_node_mask(edge_index: torch.Tensor, num_nodes: int, device) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.ones(num_nodes, device=device)
    dst_orig = edge_index[1]
    is_dst = torch.zeros(num_nodes, device=device, dtype=torch.bool)
    is_dst[dst_orig] = True
    return (~is_dst).float()

def fedg_readout(node_embeds: torch.Tensor, data: Data) -> torch.Tensor:
    H = node_embeds.size(1)
    is_cand = (data.x[:, -1] > 0.5).float().unsqueeze(1)
    end_mask = _end_node_mask(data.edge_index, data.x.size(0), data.x.device).unsqueeze(1)
    cand = global_add_pool(node_embeds * is_cand, data.batch) / global_add_pool(is_cand, data.batch).clamp_min(1.0)
    end_pool = global_add_pool(node_embeds * end_mask, data.batch) / global_add_pool(end_mask, data.batch).clamp_min(1.0)
    g_mean = global_mean_pool(node_embeds, data.batch)
    g_max = global_max_pool(node_embeds, data.batch)
    return torch.cat([cand, end_pool, g_mean, g_max], dim=1)

class DGMAAdaptActor(nn.Module):

    def __init__(self, node_dim: int=NODE_FEATURE_DIM, hidden_dim: int=64, num_layers: int=2, action_dim: int=10):
        super().__init__()
        self.encoder = FEDGEncoder(node_dim, hidden_dim, num_layers)
        context_dim = hidden_dim * 4
        self.head = nn.Sequential(nn.Linear(context_dim, 256), nn.Tanh(), nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, action_dim))

    def forward(self, data: Data) -> torch.Tensor:
        h = self.encoder(data.x, data.edge_index)
        ctx = fedg_readout(h, data)
        return self.head(ctx)

class DGMAAdaptCritic(nn.Module):

    def __init__(self, node_dim: int=NODE_FEATURE_DIM, hidden_dim: int=64, num_layers: int=2, action_dim: int=10):
        super().__init__()
        self.encoder = FEDGEncoder(node_dim, hidden_dim, num_layers)
        context_dim = hidden_dim * 4
        self.v_head = nn.Sequential(nn.Linear(context_dim, 256), nn.Tanh(), nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1))
        self.a_head = nn.Sequential(nn.Linear(context_dim, 256), nn.Tanh(), nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, action_dim))

    def forward(self, data: Data) -> torch.Tensor:
        h = self.encoder(data.x, data.edge_index)
        ctx = fedg_readout(h, data)
        v = self.v_head(ctx)
        a = self.a_head(ctx)
        q = v + (a - a.mean(dim=-1, keepdim=True))
        return q

def extract_dgma_adapt_state(ts, task_tuple, slot, task_complex_index=0):
    data, mask = encode_dag_gat_ppo_state(ts, task_tuple, slot=slot, now_time=None, task_complex_index=task_complex_index)
    return (data, mask)
