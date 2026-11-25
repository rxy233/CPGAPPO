import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class OptimizedPolicyNet(nn.Module):

    def __init__(self, n_states, n_hiddens, n_actions, feature_dim):
        super().__init__()
        self.feature_extractor = nn.Sequential(nn.Linear(feature_dim, 256), nn.Tanh(), nn.LayerNorm(256))
        self.policy_head = nn.Sequential(nn.Linear(256, n_hiddens), nn.Tanh(), nn.LayerNorm(n_hiddens), nn.Linear(n_hiddens, n_actions))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.feature_extractor(x)
        logits = self.policy_head(x)
        return logits

class OptimizedValueNet(nn.Module):

    def __init__(self, n_states, n_hiddens, feature_dim):
        super().__init__()
        self.feature_extractor = nn.Sequential(nn.Linear(feature_dim, 256), nn.Tanh(), nn.LayerNorm(256))
        self.value_head = nn.Sequential(nn.Linear(256, n_hiddens), nn.Tanh(), nn.LayerNorm(n_hiddens), nn.Linear(n_hiddens, 1))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.feature_extractor(x)
        value = self.value_head(x)
        return value

class PPO_GAE:

    def __init__(self, n_states, n_hiddens, n_actions, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device, batch_size=256, ent_coef=0.01):
        self.feature_dim = n_states
        self.n_states = n_states
        self.n_actions = n_actions
        self.device = device
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.initial_ent_coef = ent_coef
        self.epochs = epochs
        self.eps = eps
        self.gamma = gamma
        self.lmbda = lmbda
        self.actor = OptimizedPolicyNet(n_states, n_hiddens, n_actions, self.feature_dim).to(device)
        self.critic = OptimizedValueNet(n_states, n_hiddens, self.feature_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-05)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-05)
        self.actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.actor_optimizer, gamma=0.99)
        self.critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.critic_optimizer, gamma=0.99)
        self.state_cache = {}
        self.update_count = 0
        self.max_updates = 200

    def preprocess_state(self, state):
        if hasattr(state, 'x') and state.x is not None:
            x_in = state.x
        elif isinstance(state, torch.Tensor):
            x_in = state
        else:
            x_in = torch.tensor(state, dtype=torch.float32)
        if not isinstance(x_in, torch.Tensor):
            x = torch.tensor(x_in, dtype=torch.float32)
        else:
            x = x_in.clone().detach().float()
        if x.device != self.device:
            x = x.to(self.device)
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() == 2 and x.shape[0] != 1:
            if x.shape[0] > 1:
                x = x.mean(dim=0, keepdim=True)
        return x

    def take_action(self, state, task_id=None, action_mask=None, deterministic=False):
        with torch.no_grad():
            x = self.preprocess_state(state)
            logits = self.actor(x).squeeze(0)
            if torch.isnan(logits).any():
                self._reinit_network()
                logits = self.actor(x).squeeze(0)
            if action_mask is not None:
                mask = torch.tensor(action_mask, device=logits.device, dtype=torch.bool)
                if not mask.any():
                    mask = torch.ones_like(mask)
                logits = logits.masked_fill(~mask, -100000000.0)
            dist = Categorical(logits=logits)
            if deterministic:
                action = int(logits.argmax().item())
            else:
                action = int(dist.sample().item())
        return (action, None)

    def learn(self, transition_dict):
        if not transition_dict['states']:
            return (0.0, 0.0)
        state_list = [self.preprocess_state(s) for s in transition_dict['states']]
        states = torch.cat(state_list, dim=0).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.long, device=self.device).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32, device=self.device).view(-1, 1)
        masks = torch.as_tensor(np.asarray(transition_dict['masks']), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            values = self.critic(states)
            values_np = values.cpu().numpy().flatten()
            rewards_np = rewards.cpu().numpy().flatten()
            advantages_np = np.zeros_like(rewards_np)
            gae = 0.0
            next_value = 0.0
            for t in reversed(range(len(rewards_np))):
                delta = rewards_np[t] + self.gamma * next_value - values_np[t]
                gae = delta + self.gamma * self.lmbda * gae
                advantages_np[t] = gae
                next_value = values_np[t]
            advantages = torch.tensor(advantages_np, dtype=torch.float32, device=self.device).view(-1, 1)
            returns = advantages + values
            old_logits = self.actor(states)
            old_logits = torch.nan_to_num(old_logits, nan=-1000000000.0, posinf=1000000000.0, neginf=-1000000000.0)
            old_logits = old_logits.masked_fill(~masks, torch.finfo(old_logits.dtype).min)
            dist_old = Categorical(logits=old_logits)
            old_log_probs = dist_old.log_prob(actions.squeeze(1)).unsqueeze(1).detach()
        dataset_size = states.size(0)
        indices = np.arange(dataset_size)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-08)
        total_actor_loss = 0
        total_critic_loss = 0
        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]
                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]
                curr_logits = self.actor(mb_states)
                curr_logits = torch.nan_to_num(curr_logits, nan=-1000000000.0, posinf=1000000000.0, neginf=-1000000000.0)
                curr_logits = curr_logits.masked_fill(~masks[idx], torch.finfo(curr_logits.dtype).min)
                dist = Categorical(logits=curr_logits)
                curr_log_probs = dist.log_prob(mb_actions.squeeze(1)).unsqueeze(1)
                curr_entropy = dist.entropy().mean()
                ratio = torch.exp(curr_log_probs - mb_old_log_probs).squeeze(1)
                progress = self.update_count / self.max_updates
                current_eps = max(0.1, self.eps * (1.0 - progress))
                surr1 = ratio * mb_advantages.squeeze(1)
                surr2 = torch.clamp(ratio, 1 - current_eps, 1 + current_eps) * mb_advantages.squeeze(1)
                current_ent_coef = self.initial_ent_coef * (1.0 - progress)
                current_ent_coef = max(0.001, current_ent_coef)
                actor_loss = -torch.min(surr1, surr2).mean() - current_ent_coef * curr_entropy
                curr_values = self.critic(mb_states)
                critic_loss = F.mse_loss(curr_values, mb_returns)
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
        self.update_count += 1
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], 1e-06)
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], 1e-06)
        if self.update_count % 10 == 0:
            self._check_weights()
        return (total_actor_loss, total_critic_loss)

    def clear_cache(self):
        self.state_cache.clear()

    def clear_task_data(self):
        self.clear_cache()

    @property
    def training(self):
        return self.actor.training

    def _reinit_network(self):
        self.actor = OptimizedPolicyNet(self.n_states, 256, self.n_actions, self.feature_dim).to(self.device)
        self.critic = OptimizedValueNet(self.n_states, 256, self.feature_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.0001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.0005)

    def _check_weights(self):
        for param in self.actor.parameters():
            if torch.isnan(param).any():
                self._reinit_network()
                break
