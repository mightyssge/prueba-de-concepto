from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from envgen.config import load_config
from envgen.sim_engine.entities import POI, UAV

from .curriculum import CurriculumManager
from .networks import CentralCritic, GraphActor
from .ppo_agent import PPOAgent
from .spaces import ACTIONS, build_action_mask
from .train_marl import RewardWeights, make_instance, MarlEnv, _default_curriculum


@dataclass
class ScenarioInstance:
    """Deterministic validation scenario container."""

    phase_id: int
    seed: int
    data: dict


def _clone_pois(pois: Sequence[POI]) -> List[POI]:
    cloned: List[POI] = []
    for p in pois:
        poi = POI(
            y=p.y,
            x=p.x,
            dwell_ticks=p.dwell_ticks,
            priority=p.priority,
            tw=p.tw.copy() if p.tw else None,
        )
        poi.dwell_ticks_eff = getattr(p, "dwell_ticks_eff", poi.dwell_ticks)
        poi.n_persons = getattr(p, "n_persons", 0)
        poi.served = False
        poi.first_visit_t = None
        poi.served_by = None
        cloned.append(poi)
    return cloned


def clone_instance(instance: dict) -> dict:
    """Deep-copies a scenario dict to reset state between rollouts."""
    base_xy = tuple(instance["base_xy"])
    E_max = float(instance["E_max"])
    cloned_uavs: List[UAV] = []
    for src in instance["uavs"]:
        uav = UAV(uid=src.uid, pos=base_xy, E=E_max)
        uav.state = "idle"
        cloned_uavs.append(uav)
    return {
        "grid": np.array(instance["grid"], copy=True),
        "base_xy": base_xy,
        "pois": _clone_pois(instance["pois"]),
        "uavs": cloned_uavs,
        "distmap": np.array(instance["distmap"], copy=True),
        "energy_map": np.array(instance["energy_map"], copy=True),
        "E_max": E_max,
        "E_reserve": float(instance["E_reserve"]),
        "e_move_ortho": float(instance["e_move_ortho"]),
        "e_move_diag": float(instance["e_move_diag"]),
        "e_wait": float(instance["e_wait"]),
        "horizon_ticks": int(instance["horizon_ticks"]),
        "ticks_per_step": int(instance["ticks_per_step"]),
        "L_o": float(instance["L_o"]),
        "L_d": float(instance["L_d"]),
    }


def build_phase_validation_set(
    cfg: dict,
    curriculum: CurriculumManager,
    phase_id: int,
    seeds: Sequence[int],
    *,
    poi_scale: float = 1.0,
    E_max_scale: float = 1.0,
    split: str = "val",
) -> List[ScenarioInstance]:
    """Generates deterministic validation scenarios for a curriculum phase (4.4.x)."""
    if not (1 <= phase_id <= len(curriculum.phases)):
        raise ValueError(f"phase_id must be within 1..{len(curriculum.phases)}")
    phase = curriculum.phases[phase_id - 1]
    scenarios: List[ScenarioInstance] = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        instance = make_instance(
            cfg,
            phase,
            rng,
            split=split,
            poi_scale=poi_scale,
            E_max_scale=E_max_scale,
        )
        scenarios.append(ScenarioInstance(phase_id=phase_id, seed=int(seed), data=instance))
    return scenarios


def gini(values: Sequence[float]) -> float:
    """Gini coefficient (4.4.3 cooperative load metric)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    if np.allclose(arr.sum(), 0.0):
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    cum = np.cumsum(arr)
    g = (n + 1 - 2 * np.sum(cum) / cum[-1]) / n
    return float(g)


class PolicyInterface:
    """Minimal policy API for evaluation (4.4.x)."""

    def reset(self) -> None:
        raise NotImplementedError

    def select_actions(self, env: MarlEnv, observations: Dict[int, Any]) -> Dict[int, int]:
        raise NotImplementedError


class MarlActorPolicy(PolicyInterface):
    """Wraps PPOAgent for deterministic evaluation (4.4.1 & 4.4.2)."""

    def __init__(self, agent: PPOAgent):
        self.agent = agent

    def reset(self) -> None:
        pass

    def select_actions(self, env: MarlEnv, observations: Dict[int, Any]) -> Dict[int, int]:
        actions = {}
        for uid, ob in observations.items():
            act, _, _, _ = self.agent.act(
                ob.obs_vector,
                ob.node_feats,
                ob.adj_matrix,
                ob.action_mask,
                deterministic=True,
            )
            actions[uid] = act
        return actions


class GreedyBaselinePolicy(PolicyInterface):
    """Deterministic nearest-POI baseline for comparison (4.4.1)."""

    def __init__(self):
        self.targets: Dict[int, Optional[Tuple[int, int]]] = {}
        self.paths: Dict[int, List[Tuple[int, int]]] = {}
        self.dist_cache: Dict[Tuple[int, int], np.ndarray] = {}

    def reset(self) -> None:
        self.targets.clear()
        self.paths.clear()
        self.dist_cache.clear()

    def _ensure_distmap(self, env: MarlEnv, target: Tuple[int, int]) -> np.ndarray:
        if target not in self.dist_cache:
            from envgen.gridsearch import bfs_dist

            self.dist_cache[target] = bfs_dist(env.grid, target)
        return self.dist_cache[target]

    def _assign_target(self, env: MarlEnv, uav: UAV) -> None:
        best = None
        bestd = float("inf")
        for poi in env.pois:
            if poi.served:
                continue
            distmap = self._ensure_distmap(env, (poi.y, poi.x))
            d = distmap[uav.pos[0], uav.pos[1]]
            if d < bestd:
                best = (poi.y, poi.x)
                bestd = d
        self.targets[uav.uid] = best
        self.paths[uav.uid] = []
        if best is not None:
            distmap = self._ensure_distmap(env, best)
            from envgen.sim_engine.planner import greedy_follow_distmap

            self.paths[uav.uid] = greedy_follow_distmap(distmap, uav.pos, best)

    def select_actions(self, env: MarlEnv, observations: Dict[int, Any]) -> Dict[int, int]:
        actions: Dict[int, int] = {}
        for u in env.uavs:
            mask = build_action_mask(
                u,
                env.grid,
                e_move_ortho=env.e_move_ortho,
                e_move_diag=env.e_move_diag,
                e_wait=env.e_wait,
                E_reserve=env.E_reserve,
                base_xy=env.base_xy,
                pois=env.pois,
            )
            min_step = min(env.e_move_diag, env.e_move_ortho) + 1.0
            if (u.uid not in self.targets) or (self.targets[u.uid] is None):
                self._assign_target(env, u)
            tgt = self.targets.get(u.uid)
            if u.E <= env.E_reserve + min_step:
                actions[u.uid] = 10 if mask[10] else 9
                self.targets[u.uid] = env.base_xy
                self.paths[u.uid] = []
                continue
            if tgt and (u.pos == tgt):
                actions[u.uid] = 8 if mask[8] else 9
                continue
            path = self.paths.get(u.uid, [])
            if not path:
                self._assign_target(env, u)
                path = self.paths.get(u.uid, [])
            if not path:
                actions[u.uid] = 9
                continue
            ny, nx = path.pop(0)
            dy = ny - u.pos[0]
            dx = nx - u.pos[1]
            move_action = None
            for idx in range(8):
                if ACTIONS[idx] == (dy, dx):
                    move_action = idx
                    break
            if move_action is None or not mask[move_action]:
                actions[u.uid] = 9
            else:
                actions[u.uid] = move_action
        return actions


def rollout_policy(
    env: MarlEnv,
    policy: PolicyInterface,
    *,
    log_trajectories: bool = False,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Runs a single episode and returns scalar metrics + per-UAV stats (4.4.x)."""
    observations = env.observations()
    policy.reset()
    done = False
    energy_consumed = 0.0
    trajectories: Dict[int, List[Tuple[int, int]]] = {u.uid: [u.pos] for u in env.uavs} if log_trajectories else {}
    while not done:
        actions = policy.select_actions(env, observations)
        observations, _, done, info = env.step(actions)
        energy_consumed += info.get("energy_spent", 0.0)
        if log_trajectories:
            for u in env.uavs:
                trajectories[u.uid].append(u.pos)
    served_after = sum(1 for p in env.pois if p.served)
    total_pois = len(env.pois)
    coverage = served_after / max(total_pois, 1)
    violations = sum(1 for p in env.pois if getattr(p, "violated", False))
    tardiness_vals = [float(getattr(p, "tardiness", 0)) for p in env.pois]
    avg_tardiness = float(np.mean(tardiness_vals)) if tardiness_vals else 0.0
    total_dist = 0.0
    for u in env.uavs:
        total_dist += env.steps_ortho[u.uid] * env.L_o
        total_dist += env.steps_diag[u.uid] * env.L_d
    metrics = {
        "coverage": coverage,
        "violations": float(violations),
        "avg_tardiness": avg_tardiness,
        "energy_per_uav": energy_consumed / max(len(env.uavs), 1),
        "distance": total_dist,
        "rtb": float(env.rtb_count),
        "duration": float(env.tick),
        "served": float(served_after),
        "total_pois": float(total_pois),
    }
    extras = {
        "trajectories": trajectories if log_trajectories else None,
        "poi_service_counts": dict(env.served_counts),
        "distance_per_uav": {
            u.uid: env.steps_ortho[u.uid] * env.L_o + env.steps_diag[u.uid] * env.L_d for u in env.uavs
        },
        "energy_per_uav": dict(env.energy_spent),
        "action_hist": {str(k): int(v) for k, v in env.action_hist.items()},
        "rtb_events": int(env.rtb_count),
    }
    return metrics, extras


def evaluate_policy(
    cfg: dict,
    scenarios: Sequence[ScenarioInstance],
    policy_factory: Callable[[], PolicyInterface],
    *,
    policy_name: str,
    log_trajectories: bool = False,
) -> Tuple[List[Dict[str, float]], Dict[str, float], List[Dict[str, Any]]]:
    """Shared evaluation helper for 4.4.1/4.4.2/4.4.3."""
    rew = RewardWeights()
    per_episode: List[Dict[str, float]] = []
    aggregates: Dict[str, float] = {}
    extra_logs: List[Dict[str, Any]] = []
    for sc in scenarios:
        env = MarlEnv(clone_instance(sc.data), rew)
        policy = policy_factory()
        metrics, extras = rollout_policy(env, policy, log_trajectories=log_trajectories)
        metrics["phase_id"] = sc.phase_id
        metrics["seed"] = sc.seed
        per_episode.append(metrics)
        extras.update(
            {
                "policy": policy_name,
                "phase_id": sc.phase_id,
                "seed": sc.seed,
            }
        )
        extra_logs.append(extras)
    if per_episode:
        keys = [k for k in per_episode[0].keys() if k not in ("phase_id", "seed")]
        for k in keys:
            values = [m[k] for m in per_episode]
            aggregates[k + "_mean"] = float(np.mean(values))
            aggregates[k + "_std"] = float(np.std(values, ddof=0))
    return per_episode, aggregates, extra_logs


def write_metrics_csv(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_agent_for_scenarios(sample_instance: dict, checkpoint: Optional[str]) -> PPOAgent:
    """Instantiates PPO agent and loads checkpoint for evaluation (4.4.1)."""
    env = MarlEnv(clone_instance(sample_instance), RewardWeights())
    sample_obs = next(iter(env.observations().values()))
    obs_dim = sample_obs.obs_vector.size
    node_dim = sample_obs.node_feats.shape[1]
    global_dim = env.global_state().size
    actor = GraphActor(obs_dim=obs_dim, node_feat_dim=node_dim, hidden_dim=128, n_actions=11)
    critic = CentralCritic(state_dim=global_dim, hidden_dim=128)
    agent = PPOAgent(actor, critic)
    if checkpoint:
        state = torch.load(checkpoint, map_location=agent.device)
        agent.actor.load_state_dict(state["actor"])
        agent.critic.load_state_dict(state["critic"])
    return agent


def run_phase_robustness_evaluation(
    cfg: dict,
    curriculum: CurriculumManager,
    agent: PPOAgent,
    phase_id: int,
    seeds: Sequence[int],
    *,
    baseline: Optional[str] = None,
) -> List[Dict[str, float]]:
    """Evaluates MARL (and optional baseline) on a curriculum phase (4.4.2)."""
    scenarios = build_phase_validation_set(cfg, curriculum, phase_id, seeds)
    rows: List[Dict[str, float]] = []
    _, stats, _ = evaluate_policy(cfg, scenarios, lambda: MarlActorPolicy(agent), policy_name="marl")
    rows.append({"policy": "marl", "phase_id": phase_id, **stats})
    if baseline and baseline.lower() == "greedy":
        _, b_stats, _ = evaluate_policy(cfg, scenarios, lambda: GreedyBaselinePolicy(), policy_name="greedy")
        rows.append({"policy": "greedy", "phase_id": phase_id, **b_stats})
    return rows


def run_local_sensitivity(
    cfg: dict,
    curriculum: CurriculumManager,
    agent: PPOAgent,
    phase_id: int,
    seeds: Sequence[int],
    emax_factors: Sequence[float],
    poi_factors: Sequence[float],
) -> List[Dict[str, float]]:
    """Perturbs parameters around the phase defaults (4.4.2 optional sensitivity)."""
    rows: List[Dict[str, float]] = []
    for e_factor in emax_factors:
        for poi_factor in poi_factors:
            scenarios = build_phase_validation_set(
                cfg,
                curriculum,
                phase_id,
                seeds,
                poi_scale=poi_factor,
                E_max_scale=e_factor,
            )
            _, stats, _ = evaluate_policy(cfg, scenarios, lambda: MarlActorPolicy(agent), policy_name="marl")
            rows.append(
                {
                    "policy": "marl",
                    "phase_id": phase_id,
                    "E_max_factor": e_factor,
                    "poi_density_factor": poi_factor,
                    **stats,
                }
            )
    return rows


def analyze_cooperation(extra_logs: Sequence[Dict[str, Any]]) -> List[Dict[str, float]]:
    """Summarizes cooperative behavior statistics (4.4.3)."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for log in extra_logs:
        grouped.setdefault(log["policy"], []).append(log)
    rows: List[Dict[str, float]] = []
    for policy, logs in grouped.items():
        pois_per_uav = []
        ginis = []
        dist_vals = []
        energy_vals = []
        overlap_vals = []
        rtb_ratios = []
        for log in logs:
            counts = list(log["poi_service_counts"].values())
            if counts:
                pois_per_uav.append(float(np.mean(counts)))
                ginis.append(gini(counts))
            else:
                pois_per_uav.append(0.0)
                ginis.append(0.0)
            dist = list(log["distance_per_uav"].values())
            energy = list(log["energy_per_uav"].values())
            dist_vals.append(float(np.mean(dist))) if dist else dist_vals.append(0.0)
            energy_vals.append(float(np.mean(energy))) if energy else energy_vals.append(0.0)
            trajs = log.get("trajectories") or {}
            if trajs:
                uniq = [len({tuple(pt) for pt in path}) for path in trajs.values()]
                union = len(set().union(*[{tuple(pt) for pt in path} for path in trajs.values()])) if trajs else 0
                overlap = 0.0 if sum(uniq) == 0 else 1.0 - (union / max(sum(uniq), 1))
            else:
                overlap = 0.0
            overlap_vals.append(overlap)
            hist_raw = log.get("action_hist") or {}
            hist = {int(k): int(v) for k, v in hist_raw.items()}
            total_actions = sum(hist.values())
            rtb_actions = hist.get(10, 0)
            rtb_ratio = float(rtb_actions) / max(total_actions, 1) if total_actions else 0.0
            rtb_ratios.append(rtb_ratio)
        rows.append(
            {
                "policy": policy,
                "runs": len(logs),
                "avg_pois_per_uav": float(np.mean(pois_per_uav)),
                "avg_pois_per_uav_std": float(np.std(pois_per_uav, ddof=0)),
                "avg_gini": float(np.mean(ginis)),
                "avg_distance_per_uav": float(np.mean(dist_vals)),
                "avg_energy_per_uav": float(np.mean(energy_vals)),
                "avg_overlap_ratio": float(np.mean(overlap_vals)),
                "avg_rtb_action_ratio": float(np.mean(rtb_ratios)),
            }
        )
    return rows


def run_single_mode(args, cfg, curriculum):
    seeds = [int(s) for s in args.scenario_seeds.split(",")] if args.scenario_seeds else [args.seed + i for i in range(args.val_count)]
    scenarios = build_phase_validation_set(cfg, curriculum, args.val_phase, seeds)
    agent = build_agent_for_scenarios(scenarios[0].data, args.checkpoint)
    rows: List[Dict[str, float]] = []
    _, marl_stats, _ = evaluate_policy(cfg, scenarios, lambda: MarlActorPolicy(agent), policy_name="marl", log_trajectories=args.log_trajectories)
    rows.append({"policy": "marl", "phase_id": args.val_phase, **marl_stats})
    if args.baseline and args.baseline.lower() == "greedy":
        _, baseline_stats, _ = evaluate_policy(cfg, scenarios, lambda: GreedyBaselinePolicy(), policy_name="greedy", log_trajectories=args.log_trajectories)
        rows.append({"policy": "greedy", "phase_id": args.val_phase, **baseline_stats})
    if args.out_csv:
        write_metrics_csv(Path(args.out_csv), rows)
    for row in rows:
        print(f"[single] {row}")


def run_robustness_mode(args, cfg, curriculum):
    agent = None
    rows: List[Dict[str, float]] = []
    for phase_id in (1, 2, 3):
        seeds = [args.seed + i for i in range(args.robustness_count)]
        scenarios = build_phase_validation_set(cfg, curriculum, phase_id, seeds)
        if agent is None:
            agent = build_agent_for_scenarios(scenarios[0].data, args.checkpoint)
        phase_rows = run_phase_robustness_evaluation(cfg, curriculum, agent, phase_id, seeds, baseline=args.baseline)
        rows.extend(phase_rows)
    if args.robustness_out:
        write_metrics_csv(Path(args.robustness_out), rows)
    for row in rows:
        print(f"[robustness] {row}")


def run_sensitivity_mode(args, cfg, curriculum):
    seeds = [args.seed + i for i in range(args.sensitivity_count)]
    scenarios = build_phase_validation_set(cfg, curriculum, args.sensitivity_phase, seeds)
    agent = build_agent_for_scenarios(scenarios[0].data, args.checkpoint)
    emax_factors = [float(x) for x in args.sensitivity_emax.split(",")]
    poi_factors = [float(x) for x in args.sensitivity_poi.split(",")]
    rows = run_local_sensitivity(
        cfg,
        curriculum,
        agent,
        args.sensitivity_phase,
        seeds,
        emax_factors,
        poi_factors,
    )
    if args.sensitivity_out:
        write_metrics_csv(Path(args.sensitivity_out), rows)
    for row in rows:
        print(f"[sensitivity] {row}")


def run_coop_mode(args, cfg, curriculum):
    seeds = [args.seed + i for i in range(args.coop_count)]
    scenarios = build_phase_validation_set(cfg, curriculum, args.coop_phase, seeds)
    agent = build_agent_for_scenarios(scenarios[0].data, args.checkpoint)
    coop_logs: List[Dict[str, Any]] = []
    _, _, marl_logs = evaluate_policy(cfg, scenarios, lambda: MarlActorPolicy(agent), policy_name="marl", log_trajectories=True)
    coop_logs.extend(marl_logs)
    if args.baseline and args.baseline.lower() == "greedy":
        _, _, baseline_logs = evaluate_policy(cfg, scenarios, lambda: GreedyBaselinePolicy(), policy_name="greedy", log_trajectories=True)
        coop_logs.extend(baseline_logs)
    summary = analyze_cooperation(coop_logs)
    if args.coop_csv:
        write_metrics_csv(Path(args.coop_csv), summary)
    if args.coop_json:
        write_json(Path(args.coop_json), {"summary": summary, "runs": coop_logs})
    for row in summary:
        print(f"[cooperation] {row}")


def evaluate(args) -> None:
    cfg = load_config(args.config)
    curriculum = _default_curriculum()
    if args.mode == "single":
        run_single_mode(args, cfg, curriculum)
    elif args.mode == "robustness":
        run_robustness_mode(args, cfg, curriculum)
    elif args.mode == "sensitivity":
        run_sensitivity_mode(args, cfg, curriculum)
    elif args.mode == "coop":
        run_coop_mode(args, cfg, curriculum)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser("eval_marl")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, default=None, help="Path to actor/critic checkpoint")
    ap.add_argument("--mode", type=str, default="single", choices=["single", "robustness", "sensitivity", "coop"])
    ap.add_argument("--seed", type=int, default=2025, help="Base seed for validation scenario generation")
    ap.add_argument("--baseline", type=str, default=None, help="Optional baseline policy name (e.g., 'greedy')")
    ap.add_argument("--log-trajectories", action="store_true", help="Enable trajectory logging during evaluation (single mode)")
    # single-mode args
    ap.add_argument("--val-phase", type=int, default=3, choices=[1, 2, 3], help="Phase for single-mode validation")
    ap.add_argument("--val-count", type=int, default=5, help="Number of validation scenarios")
    ap.add_argument("--scenario-seeds", type=str, default=None, help="Comma-separated list of seeds for single mode")
    ap.add_argument("--out-csv", type=str, default=None, help="CSV output path for single mode")
    # robustness
    ap.add_argument("--robustness-count", type=int, default=5, help="# scenarios per phase for robustness mode")
    ap.add_argument("--robustness-out", type=str, default=None, help="CSV output path for robustness tables")
    # sensitivity
    ap.add_argument("--sensitivity-phase", type=int, default=3, choices=[1, 2, 3], help="Phase for sensitivity scans")
    ap.add_argument("--sensitivity-count", type=int, default=5, help="# scenarios per config in sensitivity mode")
    ap.add_argument("--sensitivity-emax", type=str, default="0.8,1.0,1.2", help="Comma-separated E_max factors")
    ap.add_argument("--sensitivity-poi", type=str, default="0.8,1.0,1.2", help="Comma-separated POI density factors")
    ap.add_argument("--sensitivity-out", type=str, default=None, help="CSV output path for sensitivity grids")
    # cooperative analysis
    ap.add_argument("--coop-phase", type=int, default=3, choices=[1, 2, 3], help="Phase for cooperative analysis")
    ap.add_argument("--coop-count", type=int, default=3, help="# scenarios for cooperative analysis")
    ap.add_argument("--coop-csv", type=str, default=None, help="CSV summary output for cooperative stats")
    ap.add_argument("--coop-json", type=str, default=None, help="JSON output with summary + raw logs")
    return ap.parse_args(argv)


def main():
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
