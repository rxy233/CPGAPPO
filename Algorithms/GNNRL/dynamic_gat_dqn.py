import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
BATCH_SIZE = 256
LR = 5e-05
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 50000
MASK_INVALID = -1000000000.0
TARGET_UPDATE_INTERVAL = 500

class DynamicGATNetworkAllNodes(nn.Module):

    def __init__(self, node_input_dim: int, action_dim: int, hidden_dim: int=64, heads: int=2):
        super().__init__()
        self.action_dim = action_dim
        self.gat1 = GATv2Conv(node_input_dim, hidden_dim, heads=heads, concat=False, add_self_loops=False)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim, heads=1, concat=False, add_self_loops=False)
        self.value_stream = nn.Sequential(nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        self.adv_stream = nn.Sequential(nn.Linear(hidden_dim * 2, 128), nn.ReLU(), nn.Linear(128, action_dim))

    def forward(self, x, edge_index, batch):
        h = F.elu(self.gat1(x, edge_index))
        h = F.elu(self.gat2(h, edge_index))
        g = global_mean_pool(h, batch)
        v = self.value_stream(g)
        cat_feat = torch.cat([h, g[batch]], dim=-1)
        a = self.adv_stream(cat_feat)
        return v[batch] + (a - a.mean(dim=1, keepdim=True))

class ReplayMemory:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.expert_flags = []

    def push(self, *args, is_expert=False, **kwargs):
        self.buffer.append(args)
        self.expert_flags.append(bool(is_expert))
        while len(self.expert_flags) > len(self.buffer):
            self.expert_flags.pop(0)

    def sample(self, batch_size):
        indices = random.sample(range(len(self.buffer)), min(batch_size, len(self.buffer)))
        batch = [self.buffer[i] for i in indices]
        return batch

    def __len__(self):
        return len(self.buffer)

class DynamicGAT_DQN_Agent:

    def __init__(self, node_feature_dim, action_dim, device, use_expert_data=False, expert_ratio=0.3):
        self.action_dim = action_dim
        self.device_train = device
        self.device_infer = torch.device('cpu')
        from utils.constant import para
        self.max_apps = int(para['user_num']) + 1
        self.policy_gpu = DynamicGATNetworkAllNodes(node_feature_dim, action_dim, heads=2).to(self.device_train)
        self.target_gpu = DynamicGATNetworkAllNodes(node_feature_dim, action_dim, heads=2).to(self.device_train)
        self.target_gpu.load_state_dict(self.policy_gpu.state_dict())
        self.target_gpu.eval()
        self.policy_cpu = DynamicGATNetworkAllNodes(node_feature_dim, action_dim, heads=2).to(self.device_infer)
        self.policy_cpu.load_state_dict(self.policy_gpu.state_dict())
        self.policy_cpu.eval()
        self.optimizer = optim.AdamW(self.policy_gpu.parameters(), lr=LR)
        self.memory = ReplayMemory(50000)
        self.learn_step_counter = 0
        self.steps_done = 0
        self.warmup_steps = 3000
        self.has_warmed_up = False
        self.sync_interval = 20
        self.target_sync_interval = 200
        self.use_expert_data = use_expert_data
        self.expert_ratio = expert_ratio
        self.expert_memory = ReplayMemory(5000) if use_expert_data else None

    def select_action(self, state_data, ready_mask, action_mask=None, training=True, custom_eps=None):
        x, edge_index, batch = state_data
        expected_dim = self.policy_cpu.gat1.in_channels
        if x.size(1) != expected_dim:
            raise RuntimeError(f'[select_action] x_dim mismatch: got {x.size(1)}, expected {expected_dim}\n  可能原因：get_global_graph_state_dag 中 context 被拼接了多次')
        if custom_eps is not None:
            eps_threshold = custom_eps
        elif training:
            eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1.0 * self.steps_done / EPSILON_DECAY)
            self.steps_done += 1
        else:
            eps_threshold = 0.0
        if random.random() < eps_threshold:
            ready_indices = torch.nonzero(ready_mask).squeeze()
            if ready_indices.numel() == 0:
                return (None, None)
            if ready_indices.dim() == 0:
                task_node_idx = ready_indices.item()
            else:
                task_node_idx = ready_indices[random.randint(0, len(ready_indices) - 1)].item()
            if action_mask is not None:
                if action_mask.dim() == 2:
                    task_mask = action_mask[task_node_idx]
                else:
                    task_mask = action_mask
                valid_actions = torch.nonzero(task_mask > -100000.0).squeeze()
                if valid_actions.numel() == 0:
                    offload_action = 0
                elif valid_actions.numel() == 1:
                    if valid_actions.dim() == 0:
                        offload_action = valid_actions.item()
                    else:
                        offload_action = valid_actions[0].item()
                else:
                    rand_idx = random.randint(0, len(valid_actions) - 1)
                    offload_action = valid_actions[rand_idx].item()
            else:
                offload_action = random.randint(0, self.action_dim - 1)
            return (task_node_idx, offload_action)
        with torch.no_grad():
            x, edge_index, batch = state_data
            if x.device != self.device_infer:
                x = x.to(self.device_infer)
            if edge_index.device != self.device_infer:
                edge_index = edge_index.to(self.device_infer)
            if batch.device != self.device_infer:
                batch = batch.to(self.device_infer)
            if ready_mask.device != self.device_infer:
                ready_mask = ready_mask.to(self.device_infer)
            q_values = self.policy_cpu(x, edge_index, batch)
            if ready_mask.dim() != 1:
                ready_mask = ready_mask.bool().flatten()
            ready_q_values = q_values[ready_mask]
            if ready_q_values.shape[0] == 0:
                return (None, None)
            if action_mask is not None:
                action_mask = action_mask.to(self.device_infer)
                if action_mask.dim() == 2:
                    ready_action_mask = action_mask[ready_mask]
                else:
                    ready_action_mask = action_mask.unsqueeze(0).expand(ready_q_values.shape[0], -1)
                ready_q_values = ready_q_values + ready_action_mask
            best_flat_idx = ready_q_values.argmax().item()
            best_ready_idx = best_flat_idx // self.action_dim
            offload_action = best_flat_idx % self.action_dim
            all_ready_nodes = torch.nonzero(ready_mask).squeeze()
            if all_ready_nodes.dim() == 0:
                task_node_idx = all_ready_nodes.item()
            else:
                task_node_idx = all_ready_nodes[best_ready_idx].item()
            return (task_node_idx, offload_action)

    def learn(self, updates=1):
        if not self.has_warmed_up:
            if len(self.memory) < self.warmup_steps:
                return None
            else:
                self.has_warmed_up = True
                pass
        if len(self.memory) < BATCH_SIZE:
            return None
        device = self.device_train
        transitions = self.memory.sample(BATCH_SIZE)
        data_list, next_data_list = ([], [])
        ready_list, next_ready_list = ([], [])
        mask_list, next_mask_list = ([], [])
        node_idx_list, offload_list, reward_list, done_list = ([], [], [], [])
        for t in transitions:
            state, n_idx, off, r, n_state, done, ready, n_ready, mask, n_mask = t[:10]
            x, ei, ab = state
            nx, nei, nab = n_state
            data_list.append(Data(x=x, edge_index=ei, app_batch=ab.to(torch.long)))
            next_data_list.append(Data(x=nx, edge_index=nei, app_batch=nab.to(torch.long)))
            ready_list.append(ready)
            next_ready_list.append(n_ready)
            mask_list.append(mask)
            next_mask_list.append(n_mask)
            node_idx_list.append(n_idx)
            offload_list.append(off)
            reward_list.append(r)
            done_list.append(done)
        total_loss = 0
        for _ in range(updates):
            try:
                batch_s = Batch.from_data_list(data_list).to(device)
                batch_ns = Batch.from_data_list(next_data_list).to(device)
                pool_batch_s = batch_s.batch
                pool_batch_ns = batch_ns.batch
                batch_mask = torch.cat(mask_list, dim=0).to(device)
                batch_next_mask = torch.cat(next_mask_list, dim=0).to(device)
                batch_next_ready = torch.cat(next_ready_list, dim=0).to(device)
                batch_ptr = batch_s.ptr[:-1].to(device)
                batch_node_idx = torch.tensor(node_idx_list, device=device) + batch_ptr
                batch_act = torch.tensor(offload_list, device=device)
                batch_rew = torch.tensor(reward_list, device=device)
                batch_done = torch.tensor(done_list, device=device).float()
                self.policy_gpu.train()
                q_all = self.policy_gpu(batch_s.x, batch_s.edge_index, pool_batch_s)
                current_q = q_all[batch_node_idx, batch_act]
                with torch.no_grad():
                    q_next_all = self.policy_gpu(batch_ns.x, batch_ns.edge_index, pool_batch_ns)
                    node_max_val, _ = q_next_all.max(dim=1)
                    node_max_val[~batch_next_ready] = -1000000000.0
                    node_max_val_2d = node_max_val.view(-1, 1)
                    graph_max_val_2d = global_max_pool(node_max_val_2d, pool_batch_ns)
                    graph_max_val = graph_max_val_2d.view(-1)
                    graph_max_val[graph_max_val < -100000.0] = 0.0
                    target_q_batch = batch_rew + (1.0 - batch_done) * GAMMA * graph_max_val
                loss = F.smooth_l1_loss(current_q, target_q_batch)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_gpu.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item()
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() or 'defaultcpuallocator' in str(e).lower():
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    return None
                return None
        self.learn_step_counter += 1
        if self.learn_step_counter % self.sync_interval == 0:
            self.policy_cpu.load_state_dict(self.policy_gpu.state_dict())
            self.policy_cpu.eval()
        if self.learn_step_counter % self.target_sync_interval == 0:
            self.target_gpu.load_state_dict(self.policy_gpu.state_dict())
        return total_loss / updates

    def save(self, path):
        torch.save({'policy_gpu': self.policy_gpu.state_dict()}, path)
