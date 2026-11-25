import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch_geometric.data import Batch
from .transedge_style_core import TransEdgeStyleActorCritic, NODE_FEATURE_DIM

class TransEdgeStylePPOAgent:

    def __init__(self, node_dim=NODE_FEATURE_DIM, action_dim=10, device='cpu', lr=0.0003, gamma=0.99, lmbda=0.95, eps_clip=0.2, K_epochs=4, entropy_coef=0.01):
        self.device = device
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef
        self.policy = TransEdgeStyleActorCritic(node_dim=node_dim, action_dim=action_dim).to(device)
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=1e-05)
        self.mse_loss = nn.MSELoss()
        self.memory = []
        self.inference_times = []

    def put_data(self, item):
        self.memory.append(item)

    def clear_memory(self):
        self.memory = []

    def set_episode(self, ep):
        pass

    def take_action(self, state_data, action_mask=None, deterministic=False, route_stats=None):
        start_time = time.time()
        self.policy.eval()
        with torch.no_grad():
            batch = Batch.from_data_list([state_data]).to(self.device)
            logits, _ = self.policy(batch)
            logits = logits.squeeze(0)
            if action_mask is not None:
                mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
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
        actions = torch.tensor([x[1] for x in self.memory], dtype=torch.long).to(self.device)
        rewards = [x[2] for x in self.memory]
        old_log_probs = torch.tensor([x[3] for x in self.memory], dtype=torch.float32).to(self.device)
        dones = [x[4] for x in self.memory]
        masks = [x[5] for x in self.memory]
        batch_size = 64
        values = []
        self.policy.eval()
        with torch.no_grad():
            for i in range(0, len(states), batch_size):
                b_states = Batch.from_data_list(states[i:i + batch_size]).to(self.device)
                _, v = self.policy(b_states)
                values.append(v.squeeze(-1).cpu())
        values = torch.cat(values)
        returns, advantages = ([], [])
        gae = 0
        for i in reversed(range(len(rewards))):
            next_val = 0 if i == len(rewards) - 1 else values[i + 1]
            if dones[i]:
                gae = 0
                next_val = 0
            delta = rewards[i] + self.gamma * next_val - values[i]
            gae = delta + self.gamma * self.lmbda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        if advantages.std() > 1e-06:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-06)
        total_loss = 0
        T = len(states)
        perm = np.random.permutation(T)
        self.policy.train()
        for _ in range(self.K_epochs):
            for i in range(0, T, batch_size):
                idx = perm[i:i + batch_size]
                b_states = Batch.from_data_list([states[j] for j in idx]).to(self.device)
                b_actions = actions[idx]
                b_returns = returns[idx]
                b_adv = advantages[idx]
                b_old_lp = old_log_probs[idx]
                b_masks = torch.as_tensor(np.stack([masks[j] for j in idx]), dtype=torch.bool, device=self.device)
                logits, v_pred = self.policy(b_states)
                v_pred = v_pred.squeeze(-1)
                neg_inf = torch.finfo(logits.dtype).min
                logits = logits.masked_fill(~b_masks, neg_inf)
                probs = torch.softmax(logits, dim=-1)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_log_probs - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * b_adv
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.mse_loss(v_pred, b_returns)
                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
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
