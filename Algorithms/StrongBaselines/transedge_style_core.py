import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)
from Algorithms.Baselines.gat_ppo_encoder import encode_dag_gat_ppo_state, NODE_FEATURE_DIM

class ForwardGCNEncoder(nn.Module):

    def __init__(self, input_dim=NODE_FEATURE_DIM, hidden_dim=64):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        h = self.act(self.input_proj(x))
        h1 = self.act(self.conv1(h, edge_index))
        h2 = self.act(self.conv2(h1, edge_index))
        return h2

class TransEdgeStyleActorCritic(nn.Module):

    def __init__(self, node_dim=NODE_FEATURE_DIM, hidden_dim=64, action_dim=10):
        super().__init__()
        self.encoder = ForwardGCNEncoder(node_dim, hidden_dim)
        context_dim = hidden_dim * 3
        self.actor = nn.Sequential(nn.Linear(context_dim, 256), nn.Tanh(), nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, action_dim))
        self.critic = nn.Sequential(nn.Linear(context_dim, 256), nn.Tanh(), nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, data):
        node_embeds = self.encoder(data.x, data.edge_index)
        is_target = data.x[:, -1] > 0.5
        mask_float = is_target.float().unsqueeze(1)
        candidate_embeds = global_add_pool(node_embeds * mask_float, data.batch) / global_add_pool(mask_float, data.batch).clamp_min(1.0)
        g_mean = global_mean_pool(node_embeds, data.batch)
        g_max = global_max_pool(node_embeds, data.batch)
        context = torch.cat([candidate_embeds, g_mean, g_max], dim=1)
        logits = self.actor(context)
        value = self.critic(context)
        return (logits, value)

def extract_transedge_state(ts, task_tuple, slot, task_complex_index=0):
    data, mask = encode_dag_gat_ppo_state(ts, task_tuple, slot=slot, now_time=None, task_complex_index=task_complex_index)
    return (data, mask)
