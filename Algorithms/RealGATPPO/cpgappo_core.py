import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, global_add_pool
from utils.constant import para
from Algorithms.Baselines.gat_ppo_encoder import encode_dag_gat_ppo_state

class CPGAPPOLateFusionActorCritic(nn.Module):

    def __init__(self, node_dim=27, global_dim=10, hidden_dim=64, action_dim=10, use_backward=True, use_appto_aux=False):
        super().__init__()
        self.use_backward = use_backward
        self.use_appto_aux = use_appto_aux
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        self.act = nn.ELU()
        self.fwd_conv = GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False)
        if self.use_backward:
            self.bwd_conv = GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False)
        self.global_mlp = nn.Sequential(nn.Linear(global_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, hidden_dim), nn.ELU())
        self.node_embed_dim = hidden_dim * 2 if self.use_backward else hidden_dim
        context_dim = self.node_embed_dim * 3 + hidden_dim
        self.actor = nn.Sequential(nn.Linear(context_dim, 256), nn.Tanh(), nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, action_dim))
        self.critic = nn.Sequential(nn.Linear(context_dim, 256), nn.Tanh(), nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1))
        if self.use_appto_aux:
            self.appto_head = nn.Sequential(nn.Linear(context_dim, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, data, global_feat, return_aux=False):
        x = self.act(self.input_proj(data.x))
        h_fwd = self.act(self.fwd_conv(x, data.edge_index) + x)
        if self.use_backward:
            rev_edge_index = data.edge_index[[1, 0], :]
            h_bwd = self.act(self.bwd_conv(x, rev_edge_index) + x)
            node_embeds = torch.cat([h_fwd, h_bwd], dim=-1)
        else:
            node_embeds = h_fwd
        is_target = data.x[:, -1] > 0.5
        mask_float = is_target.float().unsqueeze(1)
        target_embeds = global_add_pool(node_embeds * mask_float, data.batch) / global_add_pool(mask_float, data.batch).clamp_min(1.0)
        g_mean = global_mean_pool(node_embeds, data.batch)
        g_max = global_max_pool(node_embeds, data.batch)
        if global_feat.dim() == 1:
            global_feat = global_feat.unsqueeze(0)
        g_embed = self.global_mlp(global_feat)
        context = torch.cat([target_embeds, g_mean, g_max, g_embed], dim=1)
        actor_out = self.actor(context)
        critic_out = self.critic(context)
        if return_aux and self.use_appto_aux:
            appto_logit = self.appto_head(context).squeeze(-1)
            return (actor_out, critic_out, appto_logit)
        return (actor_out, critic_out)

def extract_cpgappo_state(ts, task_tuple, slot, task_complex_index=0):
    data, mask = encode_dag_gat_ppo_state(ts, task_tuple, slot=slot, now_time=None, task_complex_index=task_complex_index)
    data.x[:, 16:26] = 0.0
    uid = task_tuple[0]
    now = slot * para['slot_interval']
    local_q = max(0.0, float(ts.devices_exe_useful[uid]) - now)
    up_q = max(0.0, float(ts.devices_upload_useful[uid]) - now)
    edge_qs = [max(0.0, float(ts.remain_times[eid]) - now) for eid in range(para['edge_num'])]
    global_features = torch.tensor([local_q, up_q] + edge_qs, dtype=torch.float32) / 2.0
    return (data, global_features, mask)
ENERGY_WEIGHT = 0.05
DELAY_WEIGHT = 0.1

def compute_cpgappo_slack_reward(ts, uid, sid, energy, delay, enter_time, deadline_slot):
    TF = ts.finish_time[uid][sid]
    app_deadline_abs = enter_time + deadline_slot * para['slot_interval']
    slack = app_deadline_abs - TF
    r_step = -(ENERGY_WEIGHT * energy) - DELAY_WEIGHT * delay
    if slack < 0:
        r_step -= 5.0 + abs(slack) * 2.0
    elif slack < 0.2 * (app_deadline_abs - enter_time):
        r_step -= 1.0
    return (r_step, float(slack))
