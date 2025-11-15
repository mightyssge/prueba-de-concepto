from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np
import torch

from envgen.config import load_config

from .networks import CentralCritic, GraphActor
from .ppo_agent import PPOAgent
from .train_marl import RewardWeights, make_instance, MarlEnv, _default_curriculum


def evaluate(args) -> None:
    cfg = load_config(args.config)
    curriculum = _default_curriculum()
    curriculum.idx = len(curriculum.phases) - 1  # force final phase
    rng = np.random.default_rng(args.seed)
    rew = RewardWeights()

    # Build agent and load weights if provided
    sample_instance = make_instance(cfg, curriculum.current, rng, split="val")
    env = MarlEnv(sample_instance, rew)
    obs_dim = env.observations()[0].obs_vector.size
    node_dim = env.observations()[0].node_feats.shape[1]
    global_dim = env.global_state().size

    actor = GraphActor(obs_dim=obs_dim, node_feat_dim=node_dim, hidden_dim=128, n_actions=11)
    critic = CentralCritic(state_dim=global_dim, hidden_dim=128)
    agent = PPOAgent(actor, critic)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=agent.device)
        agent.actor.load_state_dict(state["actor"])
        agent.critic.load_state_dict(state["critic"])
        print(f"[eval] loaded checkpoint {args.checkpoint}")

    metrics: List[Dict[str, float]] = []

    for ep in range(args.episodes):
        if ep > 0:
            instance = make_instance(cfg, curriculum.current, rng, split="val")
            env = MarlEnv(instance, rew)

        obs = env.observations()
        done = False
        energy_consumed = 0.0
        while not done:
            actions = {}
            for uid, ob in obs.items():
                act, _, _, _ = agent.act(
                    ob.obs_vector, ob.node_feats, ob.adj_matrix, ob.action_mask, deterministic=True
                )
                actions[uid] = act
            obs, reward, done, info = env.step(actions)
            energy_consumed += info.get("energy_spent", 0.0)

        pois = env.pois
        coverage = info.get("coverage", 0.0)
        violations = sum(1 for p in pois if getattr(p, "violated", False))
        tardiness_vals = [float(getattr(p, "tardiness", 0)) for p in pois]
        avg_tardiness = float(np.mean(tardiness_vals)) if tardiness_vals else 0.0
        total_dist = 0.0
        for u in env.uavs:
            total_dist += env.steps_ortho[u.uid] * env.L_o
            total_dist += env.steps_diag[u.uid] * env.L_d

        metrics.append(
            {
                "coverage": coverage,
                "violations": violations,
                "avg_tardiness": avg_tardiness,
                "energy_per_uav": energy_consumed / max(len(env.uavs), 1),
                "distance": total_dist,
                "rtb": env.rtb_count,
                "duration": env.tick,
            }
        )
        print(
            f"[eval ep {ep:03d}] cov={coverage:.3f} viol={violations} tard={avg_tardiness:.2f} "
            f"E/uav={metrics[-1]['energy_per_uav']:.2f} dist={total_dist:.2f} rtb={env.rtb_count} dur={env.tick}"
        )

    # Aggregate
    agg = {
        k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]
    } if metrics else {}
    if agg:
        print(
            "\n[eval summary] "
            f"cov={agg['coverage']:.3f} | viol={agg['violations']:.2f} | "
            f"tard={agg['avg_tardiness']:.2f} | energy/uav={agg['energy_per_uav']:.2f} | "
            f"dist={agg['distance']:.2f} | rtb={agg['rtb']:.2f} | dur={agg['duration']:.1f}"
        )


def parse_args():
    ap = argparse.ArgumentParser("eval_marl")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, default=None, help="Path to actor/critic checkpoint")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2025)
    return ap.parse_args()


def main():
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
