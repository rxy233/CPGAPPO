import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch_geometric.data import Batch
from torch.distributions import Categorical
from .cpgappo_core import CPGAPPOLateFusionActorCritic
CP_REM_COEFF = 0.45
OVERFLOW_MULT = 1.25

def compute_guide_scores(ts, uid, task_tuple, task_complex_index, now):
    from utils.constant import para
    from Environment import computation
    import numpy as np
    edge_num = para['edge_num']
    action_dim = edge_num + 2
    guide_scores = np.full(action_dim, 1000000.0, dtype=np.float32)
    valid_mask = np.ones(action_dim, dtype=bool)
    env = ts.env
    device = env.device_list[uid]
    task_size_bytes = float(ts.get_task_size_bytes(uid, task_tuple[1]))
    local_freq = device.local_power
    upload_bw = device.local_bw if hasattr(device, 'local_bw') else np.mean(para['uplink_range'])
    _, t_local = computation.execute_consumption(task_size_bytes, local_freq, task_complex_index, 'l', local_wait=device.local_wait)
    local_queue = max(0.0, float(ts.devices_exe_useful[uid]) - now)
    guide_scores[0] = local_queue + t_local
    cloud_freq = env.cloud.cloud_power
    _, t_cloud_exec = computation.execute_consumption(task_size_bytes, cloud_freq, task_complex_index, 'c')
    nearest_edge = int(np.argmin(device.edge_distances))
    nearest_dist = float(device.edge_distances[nearest_edge])
    _, t_cloud_upload_ue = computation.upload_consumption([task_size_bytes, nearest_dist, upload_bw], 0, 'e', local_trans=device.local_trans if hasattr(device, 'local_trans') else para['local_trans'], local_wait=device.local_wait)
    _, t_cloud_upload_ec = computation.upload_consumption(task_size_bytes, 0, 'c', local_trans=device.local_trans if hasattr(device, 'local_trans') else para['local_trans'], local_wait=device.local_wait)
    upload_queue = max(0.0, float(ts.devices_upload_useful[uid]) - now)
    guide_scores[1] = upload_queue + t_cloud_upload_ue + t_cloud_upload_ec + t_cloud_exec
    for eid in range(edge_num):
        action_idx = eid + 2
        dist = device.edge_distances[eid]
        if dist == float('inf') or dist > para['edge_radius']:
            guide_scores[action_idx] = 1000000.0
            valid_mask[action_idx] = False
            continue
        edge_freq = float(env.edges[eid].edge_power * env.edges[eid].calculate_parameter)
        _, t_edge_exec = computation.execute_consumption(task_size_bytes, edge_freq, task_complex_index, 'e')
        _, t_edge_upload = computation.upload_consumption([task_size_bytes, dist, upload_bw], 0, 'e', local_trans=device.local_trans if hasattr(device, 'local_trans') else para['local_trans'], local_wait=device.local_wait)
        edge_queue = 0.0
        if eid < len(ts.core_remaining_work) and ts.core_remaining_work[eid]:
            total_work = sum(ts.core_remaining_work[eid])
            edge_queue = total_work / max(1e-09, edge_freq) if edge_freq > 0 else 1000000.0
        guide_scores[action_idx] = upload_queue + t_edge_upload + edge_queue + t_edge_exec
    return (guide_scores, valid_mask)

def compute_guide_scores_cp(ts, uid, task_tuple, task_complex_index, now):
    from utils.constant import para
    from Environment import computation
    import numpy as np
    edge_num = para['edge_num']
    action_dim = edge_num + 2
    guide_scores = np.full(action_dim, 1000000.0, dtype=np.float32)
    valid_mask = np.ones(action_dim, dtype=bool)
    env = ts.env
    device = env.device_list[uid]
    task_size_bytes = float(ts.get_task_size_bytes(uid, task_tuple[1]))
    local_freq = device.local_power
    upload_bw = device.local_bw if hasattr(device, 'local_bw') else np.mean(para['uplink_range'])
    _, t_local = computation.execute_consumption(task_size_bytes, local_freq, task_complex_index, 'l', local_wait=device.local_wait)
    local_queue = max(0.0, float(ts.devices_exe_useful[uid]) - now)
    guide_scores[0] = local_queue + t_local
    cloud_freq = env.cloud.cloud_power
    _, t_cloud_exec = computation.execute_consumption(task_size_bytes, cloud_freq, task_complex_index, 'c')
    nearest_edge = int(np.argmin(device.edge_distances))
    nearest_dist = float(device.edge_distances[nearest_edge])
    _, t_cloud_upload_ue = computation.upload_consumption([task_size_bytes, nearest_dist, upload_bw], 0, 'e', local_trans=device.local_trans if hasattr(device, 'local_trans') else para['local_trans'], local_wait=device.local_wait)
    _, t_cloud_upload_ec = computation.upload_consumption(task_size_bytes, 0, 'c', local_trans=device.local_trans if hasattr(device, 'local_trans') else para['local_trans'], local_wait=device.local_wait)
    upload_queue = max(0.0, float(ts.devices_upload_useful[uid]) - now)
    guide_scores[1] = upload_queue + t_cloud_upload_ue + t_cloud_upload_ec + t_cloud_exec
    for eid in range(edge_num):
        action_idx = eid + 2
        dist = device.edge_distances[eid]
        if dist == float('inf') or dist > para['edge_radius']:
            guide_scores[action_idx] = 1000000.0
            valid_mask[action_idx] = False
            continue
        edge_freq = float(env.edges[eid].edge_power * env.edges[eid].calculate_parameter)
        _, t_edge_exec = computation.execute_consumption(task_size_bytes, edge_freq, task_complex_index, 'e')
        _, t_edge_upload = computation.upload_consumption([task_size_bytes, dist, upload_bw], 0, 'e', local_trans=device.local_trans if hasattr(device, 'local_trans') else para['local_trans'], local_wait=device.local_wait)
        edge_queue = 0.0
        if eid < len(ts.core_remaining_work) and ts.core_remaining_work[eid]:
            total_work = sum(ts.core_remaining_work[eid])
            edge_queue = total_work / max(1e-09, edge_freq) if edge_freq > 0 else 1000000.0
        guide_scores[action_idx] = upload_queue + t_edge_upload + edge_queue + t_edge_exec
    sid = int(task_tuple[1])
    g = ts.subgraph_list[uid].nx_graph
    try:
        succs = list(g.successors(sid))
    except Exception:
        succs = []
    if not succs:
        cp_depth = 0
    else:

        def _cp_depth_from(node, memo):
            if node in memo:
                return memo[node]
            try:
                node_succs = list(g.successors(node))
            except Exception:
                memo[node] = 0
                return 0
            if not node_succs:
                memo[node] = 0
                return 0
            depth = 1 + max((_cp_depth_from(s, memo) for s in node_succs))
            memo[node] = depth
            return depth
        cp_depth = _cp_depth_from(sid, {})
    app_deadline_slot = ts.get_app_deadline_slot(uid) if hasattr(ts, 'get_app_deadline_slot') else para['deadline_slot']
    if uid < len(ts.enter_time) and ts.enter_time[uid] != float('inf'):
        enter_time = float(ts.enter_time[uid])
    else:
        enter_time = float(now)
    app_deadline_abs = enter_time + app_deadline_slot * para['slot_interval']
    cp_factor = min(1.0, cp_depth / 4.0)
    for a in range(action_dim):
        if not valid_mask[a]:
            continue
        immediate_cost = float(guide_scores[a])
        pred_finish = now + immediate_cost
        time_left_after = max(0.0, app_deadline_abs - now) - immediate_cost
        cp_remain_est = cp_factor * app_deadline_slot * para['slot_interval'] * CP_REM_COEFF
        downstream_overflow = max(0.0, cp_remain_est - time_left_after)
        guide_scores[a] = immediate_cost + OVERFLOW_MULT * downstream_overflow
    return (guide_scores, valid_mask)

class GAT_PPO_Agent_CPGAPPO:

    def __init__(self, node_dim=27, global_dim=10, action_dim=10, device='cpu', lr=0.0003, gamma=0.99, lmbda=0.95, eps_clip=0.2, K_epochs=4, entropy_coef=0.02, lambda_guide=0.1, use_backward=True, use_appto_aux=False, aux_loss_coef=0.1):
        self.device = device
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef
        self.lambda_guide = lambda_guide
        self.use_appto_aux = use_appto_aux
        self.aux_loss_coef = aux_loss_coef
        self.policy = CPGAPPOLateFusionActorCritic(node_dim, global_dim, 64, action_dim, use_backward=use_backward, use_appto_aux=use_appto_aux).to(device)
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=1e-05)
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        self.memory = []
        self.inference_times = []

    def put_data(self, item):
        self.memory.append(item)

    def clear_memory(self):
        self.memory = []

    def take_action(self, state_data, global_feat, action_mask=None, deterministic=False):
        start_time = time.time()
        self.policy.eval()
        with torch.no_grad():
            batch = Batch.from_data_list([state_data]).to(self.device)
            g_feat = global_feat.to(self.device)
            logits, _ = self.policy(batch, g_feat)
            logits = logits.squeeze(0)
            if action_mask is not None:
                mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
                if not mask_t.any():
                    raise ValueError('A task with no feasible placement must be deferred')
                neg_inf = torch.finfo(logits.dtype).min
                logits = logits.masked_fill(~mask_t, neg_inf)
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)
            if deterministic:
                action = torch.argmax(probs).item()
            else:
                action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action, device=self.device)).item()
        self.inference_times.append((time.time() - start_time) * 1000)
        return (action, log_prob)

    def update(self):
        if not self.memory:
            return 0.0
        states = [x[0] for x in self.memory]
        global_feats = torch.stack([x[1] for x in self.memory]).to(self.device)
        actions = torch.tensor([x[2] for x in self.memory], dtype=torch.long).to(self.device)
        rewards = [x[3] for x in self.memory]
        old_log_probs = torch.tensor([x[4] for x in self.memory], dtype=torch.float32).to(self.device)
        masks = [x[6] for x in self.memory]
        guide_actions = [x[7] for x in self.memory]
        batch_size = 64
        values = []
        self.policy.eval()
        with torch.no_grad():
            for i in range(0, len(states), batch_size):
                b_states = Batch.from_data_list(states[i:i + batch_size]).to(self.device)
                b_gfeats = global_feats[i:i + batch_size]
                _, v = self.policy(b_states, b_gfeats)
                values.append(v.squeeze(-1).cpu())
        values = torch.cat(values)
        returns, advantages, gae = ([], [], 0)
        for i in reversed(range(len(rewards))):
            next_val = 0 if i == len(rewards) - 1 else values[i + 1]
            if self.memory[i][5]:
                gae = 0
                next_val = 0
            delta = rewards[i] + self.gamma * next_val - values[i]
            gae = delta + self.gamma * self.lmbda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        if len(advantages) > 1 and advantages.std() > 1e-06:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-06)
        guide_actions_t = torch.tensor(guide_actions, dtype=torch.long).to(self.device)
        self.policy.train()
        total_loss = 0
        T = len(states)
        for _ in range(self.K_epochs):
            perm = np.random.permutation(T)
            for i in range(0, T, batch_size):
                idx = perm[i:i + batch_size]
                b_states = Batch.from_data_list([states[j] for j in idx]).to(self.device)
                b_gfeats = global_feats[idx]
                b_actions = actions[idx]
                b_returns = returns[idx]
                b_adv = advantages[idx]
                b_old_lp = old_log_probs[idx]
                b_masks = torch.as_tensor(np.stack([masks[j] for j in idx]), dtype=torch.bool, device=self.device)
                b_guide = guide_actions_t[idx]
                logits, v_pred = self.policy(b_states, b_gfeats) if not self.use_appto_aux else (None, None)
                if self.use_appto_aux:
                    logits, v_pred, appto_logit = self.policy(b_states, b_gfeats, return_aux=True)
                v_pred = v_pred.squeeze(-1)
                neg_inf = torch.finfo(logits.dtype).min
                masked_logits = logits.masked_fill(~b_masks, neg_inf)
                probs = torch.softmax(masked_logits, dim=-1)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_log_probs - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * b_adv
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.mse_loss(v_pred, b_returns)
                guide_ce = self.ce_loss(masked_logits, b_guide) if self.lambda_guide > 0 else logits.new_zeros(())
                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy + self.lambda_guide * guide_ce
                if self.use_appto_aux:
                    b_appto_labels = torch.tensor([float(self.memory[j][8]) for j in idx], dtype=torch.float32, device=self.device)
                    aux_loss = F.binary_cross_entropy_with_logits(appto_logit, b_appto_labels)
                    loss = loss + self.aux_loss_coef * aux_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()
        return total_loss

    def print_inference_stats(self):
        if not self.inference_times:
            return
        arr = np.array(self.inference_times)
