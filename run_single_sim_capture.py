import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from envgen.config import load_config  # noqa: E402
from marl.envs import (  # noqa: E402
    build_lurigancho_fixed_episode,
    load_lurigancho_fixed_data,
    load_lurigancho_map,
)
from marl.eval_marl import clone_instance, MarlActorPolicy  # noqa: E402
from marl.networks import CentralCritic, GraphActor  # noqa: E402
from marl.ppo_agent import PPOAgent  # noqa: E402
from marl.train_marl import RewardWeights, MarlEnv  # noqa: E402


CONFIG_PATH = Path("config.json")
SCENARIO_PATH = Path("lurigancho_scenario.json")
POIS_PATH = Path("lurigancho_pois_val.json")
CHECKPOINT_PATH = Path("results/ckpt_step3_lurigancho_fixed.pt")
VIDEO_PATH = Path("results/lurigancho_single_run.gif")
SIM_SEED = 2025


def compute_status(env: MarlEnv) -> dict:
    total = len(env.pois)
    served = sum(1 for p in env.pois if p.served)
    coverage = served / max(total, 1)
    violations = sum(1 for p in env.pois if getattr(p, "violated", False))
    avg_energy = float(np.mean([u.E for u in env.uavs])) if env.uavs else 0.0
    return {
        "total_pois": total,
        "served": served,
        "coverage": coverage,
        "violations": violations,
        "avg_energy": avg_energy,
    }


def render_env_frame(env: MarlEnv, trajectories: dict[int, list[tuple[int, int]]], step_idx: int) -> np.ndarray:
    status = compute_status(env)
    H, W = env.grid.shape
    bg = np.where(env.grid, 0.25, 0.95)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(bg, cmap="gray", origin="upper")

    pending = [(p.x, p.y) for p in env.pois if not p.served]
    served = [(p.x, p.y) for p in env.pois if p.served]
    if pending:
        ax.scatter([x for x, _ in pending], [y for _, y in pending], s=20, c="tomato", edgecolors="black", linewidths=0.3)
    if served:
        ax.scatter([x for x, _ in served], [y for _, y in served], s=20, c="mediumseagreen", edgecolors="black", linewidths=0.3)
    by, bx = env.base_xy
    ax.scatter([bx], [by], marker="s", s=80, facecolors="none", edgecolors="deepskyblue", linewidths=2.0)
    colors = plt.cm.tab10.colors
    for idx, u in enumerate(env.uavs):
        y, x = u.pos
        ax.scatter([x], [y], marker="^", s=120, facecolors=colors[idx % len(colors)], edgecolors="black", linewidths=0.5)
        if trajectories and u.uid in trajectories:
            xs = [pos[1] for pos in trajectories[u.uid]]
            ys = [pos[0] for pos in trajectories[u.uid]]
            ax.plot(xs, ys, color=colors[idx % len(colors)], linewidth=1.0, alpha=0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_title(f"Paso {step_idx} | tick {env.tick} | cobertura {status['coverage']:.1%}")
    fig.tight_layout()
    canvas = fig.canvas
    canvas.draw()
    frame = np.asarray(canvas.buffer_rgba())
    frame = frame[:, :, :3].copy()
    plt.close(fig)
    return frame


def run_episode(env: MarlEnv, policy: MarlActorPolicy, log_interval: int = 10):
    observations = env.observations()
    policy.reset()
    done = False
    step_idx = 0
    frames: list[np.ndarray] = []
    trajectories = {u.uid: [u.pos] for u in env.uavs}
    while not done:
        frames.append(render_env_frame(env, trajectories, step_idx))
        actions = policy.select_actions(env, observations)
        observations, _, done, _ = env.step(actions)
        step_idx += 1
        for u in env.uavs:
            trajectories[u.uid].append(u.pos)
        if step_idx % log_interval == 0 or done:
            status = compute_status(env)
            print(
                f"[paso {step_idx:03d}] tick={env.tick:05d} "
                f"serv={status['served']}/{status['total_pois']} "
                f"cov={status['coverage']:.1%} energía_prom={status['avg_energy']:.1f} "
                f"rtb={env.rtb_count}"
            )
    frames.append(render_env_frame(env, trajectories, step_idx))
    return frames


def summarize_episode(env: MarlEnv) -> dict:
    served_after = sum(1 for p in env.pois if p.served)
    total_pois = max(len(env.pois), 1)
    coverage = served_after / total_pois
    violations = sum(1 for p in env.pois if getattr(p, "violated", False))
    tardiness_vals = [float(getattr(p, "tardiness", 0.0)) for p in env.pois]
    avg_tardiness = float(np.mean(tardiness_vals)) if tardiness_vals else 0.0
    energy_tot = sum(env.energy_spent.values())
    energy_per_uav = energy_tot / max(len(env.uavs), 1)
    distance_total = sum(
        env.steps_ortho[u.uid] * env.L_o + env.steps_diag[u.uid] * env.L_d
        for u in env.uavs
    )
    return {
        "coverage": coverage,
        "served": served_after,
        "violations": float(violations),
        "avg_tardiness": avg_tardiness,
        "energy_per_uav": energy_per_uav,
        "distance_total_m": distance_total,
        "duration_ticks": float(env.tick),
        "rtb_events": int(env.rtb_count),
    }


def main():
    cfg = load_config(CONFIG_PATH)
    map_data = load_lurigancho_map(SCENARIO_PATH)
    fixed_data = load_lurigancho_fixed_data(POIS_PATH)
    rng = np.random.default_rng(SIM_SEED)
    episode, ei_map, hooks = build_lurigancho_fixed_episode(map_data, fixed_data, cfg, rng, split="val")

    # Build agent the same way eval_lurigancho.py does
    dim_env = MarlEnv(
        clone_instance(episode),
        RewardWeights(),
        global_obs=ei_map,
        hooks=hooks,
        env_mode="lurigancho_fixed",
        ignore_horizon=True,
    )
    sample_obs = next(iter(dim_env.observations().values()))
    obs_dim = sample_obs.obs_vector.size
    node_dim = sample_obs.node_feats.shape[1]
    global_dim = dim_env.global_state().size
    actor = GraphActor(obs_dim=obs_dim, node_feat_dim=node_dim, hidden_dim=128, n_actions=11)
    critic = CentralCritic(state_dim=global_dim, hidden_dim=128)
    agent = PPOAgent(actor, critic)
    state = torch.load(CHECKPOINT_PATH, map_location=agent.device)
    agent.actor.load_state_dict(state["actor"])
    agent.critic.load_state_dict(state["critic"])
    policy = MarlActorPolicy(agent, deterministic=True)

    clone = clone_instance(episode)
    poi_cells = [p.y * map_data.cols + p.x for p in clone["pois"]]
    ei_map.reset_dynamic(poi_cells)
    env = MarlEnv(
        clone,
        RewardWeights(),
        global_obs=ei_map,
        hooks=hooks,
        env_mode="lurigancho_fixed",
        ignore_horizon=True,
    )

    frames = run_episode(env, policy, log_interval=10)
    metrics = summarize_episode(env)

    VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(VIDEO_PATH, frames, duration=0.45)
    print(f"[video] guardado en {VIDEO_PATH}")
    print("[stats]", metrics)


if __name__ == "__main__":
    main()
