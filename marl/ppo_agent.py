from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .networks import CentralCritic, GraphActor


def _pad_nodes(
    feats_list: List[np.ndarray], adj_list: List[np.ndarray]
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_nodes = max(f.shape[0] for f in feats_list)
    padded_feats, padded_adj = [], []
    for feats, adj in zip(feats_list, adj_list):
        pad_n = max_nodes - feats.shape[0]
        if pad_n > 0:
            feats = np.pad(feats, ((0, pad_n), (0, 0)), constant_values=0.0)
            adj = np.pad(adj, ((0, pad_n), (0, pad_n)), constant_values=0.0)
        padded_feats.append(torch.tensor(feats, dtype=torch.float32))
        padded_adj.append(torch.tensor(adj, dtype=torch.float32))
    return torch.stack(padded_feats, dim=0), torch.stack(padded_adj, dim=0)


@dataclass
class Transition:
    obs_vector: np.ndarray
    node_feats: np.ndarray
    adj: np.ndarray
    action_mask: np.ndarray
    action: int
    logprob: float
    reward: float
    value: float
    next_value: float
    done: bool
    global_state: np.ndarray


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def add(self, tr: Transition) -> None:
        self.data.append(tr)

    def clear(self) -> None:
        self.data: List[Transition] = []
        self.advantages: Optional[torch.Tensor] = None
        self.returns: Optional[torch.Tensor] = None

    def compute_advantages(self, gamma: float, lam: float) -> None:
        rewards = [t.reward for t in self.data]
        values = [t.value for t in self.data]
        next_values = [t.next_value for t in self.data]
        dones = [t.done for t in self.data]
        gae = 0.0
        adv_list = [0.0 for _ in rewards]
        for i in reversed(range(len(rewards))):
            next_value = next_values[i] if i < len(next_values) else 0.0
            non_terminal = 0.0 if dones[i] else 1.0
            delta = rewards[i] + gamma * next_value * non_terminal - values[i]
            gae = delta + gamma * lam * non_terminal * gae
            adv_list[i] = gae
        values_t = torch.tensor(values, dtype=torch.float32)
        self.advantages = torch.tensor(adv_list, dtype=torch.float32)
        self.returns = self.advantages + values_t

    def as_tensors(self) -> Tuple[torch.Tensor, ...]:
        obs_arr = np.stack([t.obs_vector for t in self.data], axis=0).astype(np.float32)
        obs_vec = torch.from_numpy(obs_arr)
        actions_arr = np.asarray([t.action for t in self.data], dtype=np.int64)
        actions = torch.from_numpy(actions_arr)
        logprobs_arr = np.asarray([t.logprob for t in self.data], dtype=np.float32)
        logprobs = torch.from_numpy(logprobs_arr)
        masks_arr = np.stack([t.action_mask for t in self.data], axis=0).astype(bool)
        masks = torch.from_numpy(masks_arr)
        states_arr = np.stack([t.global_state for t in self.data], axis=0).astype(np.float32)
        states = torch.from_numpy(states_arr)
        node_feats, adjs = _pad_nodes(
            [t.node_feats for t in self.data], [t.adj for t in self.data]
        )
        return obs_vec, node_feats, adjs, masks, actions, logprobs, states


class PPOAgent:
    def __init__(
        self,
        actor: GraphActor,
        critic: CentralCritic,
        *,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lam = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.buffer = RolloutBuffer()

    def _mask_logits(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        neg_inf = torch.finfo(logits.dtype).min
        return logits.masked_fill(~mask, neg_inf)

    def act(
        self,
        obs_vec: np.ndarray,
        node_feats: np.ndarray,
        adj: np.ndarray,
        action_mask: np.ndarray,
        *,
        hidden: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[int, float, float, Optional[torch.Tensor]]:
        obs_t = torch.tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        node_t = torch.tensor(node_feats, dtype=torch.float32, device=self.device).unsqueeze(0)
        adj_t = torch.tensor(adj, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        logits, next_hidden, _ = self.actor(obs_t, node_t, adj_t, hidden)
        masked_logits = self._mask_logits(logits, mask_t)
        dist = Categorical(logits=masked_logits)
        if deterministic:
            action = int(torch.argmax(masked_logits, dim=-1).item())
        else:
            action = int(dist.sample().item())
        logprob = float(dist.log_prob(torch.tensor(action, device=self.device)).item())
        return action, logprob, float(0.0), next_hidden

    def store_transition(self, tr: Transition) -> None:
        self.buffer.add(tr)

    def update(self, batch_size: int = 64, epochs: int = 4) -> dict:
        if self.buffer.advantages is None or self.buffer.returns is None:
            self.buffer.compute_advantages(self.gamma, self.lam)

        obs_vec, node_feats, adjs, masks, actions, old_logprobs, states = self.buffer.as_tensors()
        adv = self.buffer.advantages.clone()
        ret = self.buffer.returns.clone()

        obs_vec = obs_vec.to(self.device)
        node_feats = node_feats.to(self.device)
        adjs = adjs.to(self.device)
        masks = masks.to(self.device)
        actions = actions.to(self.device)
        old_logprobs = old_logprobs.to(self.device)
        adv = adv.to(self.device)
        ret = ret.to(self.device)
        states = states.to(self.device)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = obs_vec.size(0)
        idxs = torch.arange(n)

        stats = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}
        for _ in range(epochs):
            perm = idxs[torch.randperm(n)]
            for start in range(0, n, batch_size):
                batch = perm[start : start + batch_size]
                logits, _, _ = self.actor(obs_vec[batch], node_feats[batch], adjs[batch])
                masked_logits = self._mask_logits(logits, masks[batch])
                dist = Categorical(logits=masked_logits)
                logprob = dist.log_prob(actions[batch])
                entropy = dist.entropy().mean()

                ratios = torch.exp(logprob - old_logprobs[batch])
                surr1 = ratios * adv[batch]
                surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv[batch]
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                value_pred = self.critic(states[batch])
                critic_loss = F.mse_loss(value_pred, ret[batch])

                loss = actor_loss + self.value_coef * critic_loss

                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), self.max_grad_norm)
                self.actor_opt.step()
                self.critic_opt.step()

                stats["actor_loss"] = float(actor_loss.item())
                stats["critic_loss"] = float(critic_loss.item())
                stats["entropy"] = float(entropy.item())

        self.buffer.clear()
        return stats
