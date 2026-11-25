import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from typing import List, Optional
_cur_dir = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.dirname(os.path.dirname(_cur_dir))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)
from Algorithms.StrongBaselines.dgma_adapt_core import DGMAAdaptActor, DGMAAdaptCritic, NODE_FEATURE_DIM

class PERBuffer:

    def __init__(self, capacity: int, alpha: float=0.6):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.data: List[Optional[tuple]] = [None] * self.capacity
        self.priorities = np.zeros(self.capacity, dtype=np.float64)
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0

    def __len__(self):
        return self.size

    def push(self, transition: tuple, priority: Optional[float]=None):
        if priority is None:
            priority = self.max_priority
        self.data[self.pos] = transition
        self.priorities[self.pos] = float(priority) ** self.alpha
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float=0.4):
        if self.size == 0:
            return ([], np.array([]), np.array([]))
        p = self.priorities[:self.size]
        total = p.sum()
        if total <= 0:
            probs = np.full(self.size, 1.0 / self.size)
        else:
            probs = p / total
        indices = np.random.choice(self.size, batch_size, replace=True, p=probs)
        samples = [self.data[i] for i in indices]
        weights = (self.size * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        return (samples, indices, weights.astype(np.float32))

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray, eps: float=1e-05):
        pr = (np.abs(td_errors) + eps) ** self.alpha
        self.priorities[indices] = pr
        self.max_priority = max(self.max_priority, float(pr.max(initial=0.0)))

class DGMAAdaptAgent:

    def __init__(self, node_dim: int=NODE_FEATURE_DIM, action_dim: int=10, device: str='cpu', hidden_dim: int=64, num_layers: int=2, lr_actor: float=0.0003, lr_critic: float=0.0003, gamma: float=0.99, tau: float=0.005, gumbel_tau: float=1.0, batch_size: int=64, buffer_capacity: int=20000, per_alpha: float=0.6, per_beta0: float=0.4, per_beta_anneal_steps: int=20000, updates_per_step: int=1, min_updates_after: int=256, grad_clip: float=1.0):
        self.device = device
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.gumbel_tau = gumbel_tau
        self.batch_size = batch_size
        self.updates_per_step = updates_per_step
        self.min_updates_after = min_updates_after
        self.grad_clip = grad_clip
        self.actor = DGMAAdaptActor(node_dim, hidden_dim, num_layers, action_dim).to(device)
        self.critic = DGMAAdaptCritic(node_dim, hidden_dim, num_layers, action_dim).to(device)
        self.actor_target = DGMAAdaptActor(node_dim, hidden_dim, num_layers, action_dim).to(device)
        self.critic_target = DGMAAdaptCritic(node_dim, hidden_dim, num_layers, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.actor_target.parameters():
            p.requires_grad = False
        for p in self.critic_target.parameters():
            p.requires_grad = False
        self.opt_actor = torch.optim.AdamW(self.actor.parameters(), lr=lr_actor, weight_decay=1e-05)
        self.opt_critic = torch.optim.AdamW(self.critic.parameters(), lr=lr_critic, weight_decay=1e-05)
        self.buffer = PERBuffer(buffer_capacity, alpha=per_alpha)
        self.per_beta0 = per_beta0
        self.per_beta_anneal_steps = per_beta_anneal_steps
        self.global_step = 0
        self.inference_times: List[float] = []
        self.last_actor_loss = 0.0
        self.last_critic_loss = 0.0

    @property
    def policy(self):
        return self.actor

    def set_episode(self, ep: int):
        pass

    @torch.no_grad()
    def take_action(self, state_data, action_mask=None, deterministic: bool=False, route_stats=None):
        t0 = time.time()
        self.actor.eval()
        batch = Batch.from_data_list([state_data]).to(self.device)
        logits = self.actor(batch).squeeze(0)
        if action_mask is not None:
            mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
            neg_inf = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(~mask_t, neg_inf)
        if deterministic:
            action = int(torch.argmax(logits).item())
        else:
            probs = torch.softmax(logits, dim=-1)
            action = int(torch.distributions.Categorical(probs=probs).sample().item())
        log_prob = F.log_softmax(logits, dim=-1)[action].item()
        self.inference_times.append((time.time() - t0) * 1000.0)
        return (action, log_prob)

    def store(self, s, a, r, s_next, done, mask, next_mask):
        self.buffer.push((s, int(a), float(r), s_next, bool(done), list(mask), list(next_mask) if next_mask is not None else [1] * self.action_dim))

    def _beta(self):
        frac = min(1.0, self.global_step / max(1, self.per_beta_anneal_steps))
        return self.per_beta0 + frac * (1.0 - self.per_beta0)

    def _soft_update(self, target: nn.Module, source: nn.Module):
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * sp.data)

    def update(self):
        if len(self.buffer) < max(self.batch_size, self.min_updates_after):
            return (0.0, 0.0)
        self.global_step += 1
        samples, indices, is_weights = self.buffer.sample(self.batch_size, beta=self._beta())
        states = [t[0] for t in samples]
        actions = torch.tensor([t[1] for t in samples], dtype=torch.long, device=self.device)
        rewards = torch.tensor([t[2] for t in samples], dtype=torch.float32, device=self.device)
        next_states = [t[3] if t[3] is not None else t[0] for t in samples]
        dones = torch.tensor([float(t[4]) for t in samples], dtype=torch.float32, device=self.device)
        masks = torch.tensor(np.array([t[5] for t in samples]), dtype=torch.bool, device=self.device)
        next_masks = torch.tensor(np.array([t[6] for t in samples]), dtype=torch.bool, device=self.device)
        is_weights_t = torch.tensor(is_weights, dtype=torch.float32, device=self.device)
        batch_s = Batch.from_data_list(states).to(self.device)
        batch_sp = Batch.from_data_list(next_states).to(self.device)
        self.critic.train()
        q_all = self.critic(batch_s)
        q_sa = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next_all = self.critic_target(batch_sp)
            neg_inf = torch.finfo(q_next_all.dtype).min
            q_next_masked = q_next_all.masked_fill(~next_masks, neg_inf)
            a_logits_next = self.actor_target(batch_sp)
            a_logits_next = a_logits_next.masked_fill(~next_masks, neg_inf)
            a_next = a_logits_next.argmax(dim=-1, keepdim=True)
            q_next = q_next_masked.gather(1, a_next).squeeze(1)
            y = rewards + self.gamma * (1.0 - dones) * q_next
        td_err = (y - q_sa).detach()
        critic_loss = (is_weights_t * (q_sa - y).pow(2)).mean()
        self.opt_critic.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.opt_critic.step()
        self.actor.train()
        logits = self.actor(batch_s)
        logits_masked = logits.masked_fill(~masks, torch.finfo(logits.dtype).min)
        soft_one_hot = F.gumbel_softmax(logits_masked, tau=self.gumbel_tau, hard=False, dim=-1)
        q_weighted = (soft_one_hot * q_all.detach()).sum(dim=-1)
        actor_loss = -(is_weights_t * q_weighted).mean()
        self.opt_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.opt_actor.step()
        self.buffer.update_priorities(indices, td_err.detach().cpu().numpy())
        self._soft_update(self.critic_target, self.critic)
        self._soft_update(self.actor_target, self.actor)
        self.last_actor_loss = float(actor_loss.item())
        self.last_critic_loss = float(critic_loss.item())
        return (self.last_actor_loss, self.last_critic_loss)

    def print_inference_stats(self):
        if not self.inference_times:
            return
        arr = np.array(self.inference_times)

    def state_dict(self):
        return {'actor': self.actor.state_dict(), 'critic': self.critic.state_dict(), 'actor_target': self.actor_target.state_dict(), 'critic_target': self.critic_target.state_dict()}

    def load_actor_state_dict(self, sd):
        self.actor.load_state_dict(sd)
