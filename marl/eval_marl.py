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
from envgen.gridsearch import bfs_dist
from envgen.sim_engine.entities import POI, UAV
from envgen.sim_engine.planner import greedy_follow_distmap

from .curriculum import CurriculumManager
from .networks import CentralCritic, GraphActor
from .ppo_agent import PPOAgent
from .envs import (
    load_lurigancho_map,
    load_lurigancho_fixed_data,
    build_lurigancho_random_episode,
    build_lurigancho_fixed_episode,
)
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

    def __init__(self, agent: PPOAgent, *, deterministic: bool = True):
        self.agent = agent
        self.deterministic = deterministic

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
                deterministic=self.deterministic,
            )
            actions[uid] = act
        return actions


class GuidedMarlPolicy(PolicyInterface):
    """Marl policy with simple RTB guidance (energía + estancamiento)."""

    def __init__(self, agent: PPOAgent, *, deterministic: bool = True, energy_margin: float = 5.0, stagnation_steps: int = 200, disabled: bool = False):
        self.agent = agent
        self.deterministic = deterministic
        self.energy_margin = max(0.0, energy_margin)
        self.stagnation_steps = stagnation_steps
        self.disabled = disabled
        self.reset()

    def reset(self) -> None:
        self.stagnation_counter = 0
        self.force_rtb = False
        self.served_count = None
        self.violated_count = None

    def _energy_needed(self, env: MarlEnv, uav: UAV) -> float:
        base_cost = float(env.energy_map[uav.pos[0], uav.pos[1]]) if hasattr(env, "energy_map") else 0.0
        return base_cost + env.E_reserve + self.energy_margin

    def select_actions(self, env: MarlEnv, observations: Dict[int, Any]) -> Dict[int, int]:
        uid_to_uav = {u.uid: u for u in env.uavs}
        if self.served_count is None:
            self.served_count = sum(1 for p in env.pois if p.served)
        if self.violated_count is None:
            self.violated_count = sum(1 for p in env.pois if getattr(p, "violated", False))
        actions: Dict[int, int] = {}
        for uid, ob in observations.items():
            forced_action = None
            if not self.disabled:
                mask = getattr(ob, "action_mask", None)
                can_rtb = bool(mask[10]) if mask is not None and len(mask) > 10 else False
                if self.force_rtb and can_rtb:
                    forced_action = 10
                else:
                    u = uid_to_uav[uid]
                    energy_needed = self._energy_needed(env, u)
                    if np.isfinite(energy_needed) and can_rtb and u.E <= energy_needed:
                        forced_action = 10
            act, _, _, _ = self.agent.act(
                ob.obs_vector,
                ob.node_feats,
                ob.adj_matrix,
                ob.action_mask,
                deterministic=self.deterministic,
                forced_action=forced_action,
            )
            actions[uid] = act
        return actions

    def post_step(self, env: MarlEnv) -> None:
        served_after = sum(1 for p in env.pois if p.served)
        violated_after = sum(1 for p in env.pois if getattr(p, "violated", False))
        if served_after > (self.served_count or 0) or violated_after > (self.violated_count or 0):
            self.stagnation_counter = 0
            self.force_rtb = False
        else:
            self.stagnation_counter += 1
            if self.stagnation_steps > 0 and self.stagnation_counter >= self.stagnation_steps:
                self.force_rtb = True
                self.stagnation_counter = 0
        self.served_count = served_after
        self.violated_count = violated_after


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


class GeneticBaselinePolicy(PolicyInterface):
    """Simple GA-based planner that assigns static POI routes to the UAV fleet."""

    def __init__(self, pop_size: int = 32, generations: int = 40, mutation_prob: float = 0.2):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_prob = mutation_prob
        self.rng = np.random.default_rng(1234)
        self.routes: Dict[int, List[Tuple[int, int]]] = {}
        self.current_targets: Dict[int, Optional[Tuple[int, int]]] = {}
        self.paths: Dict[int, List[Tuple[int, int]]] = {}
        self.poi_lookup: Dict[Tuple[int, int], POI] = {}
        self.dist_cache: Dict[Tuple[int, int], np.ndarray] = {}

    def reset(self) -> None:
        self.routes = {}
        self.current_targets = {}
        self.paths = {}
        self.poi_lookup = {}
        self.dist_cache = {}

    def select_actions(self, env: MarlEnv, observations: Dict[int, Any]) -> Dict[int, int]:
        if not self.routes:
            self._plan_routes(env)
        actions: Dict[int, int] = {}
        for u in env.uavs:
            actions[u.uid] = self._next_action(env, u)
        return actions

    def _plan_routes(self, env: MarlEnv) -> None:
        self.poi_lookup = {(p.y, p.x): p for p in env.pois}
        remaining_pois = [p for p in env.pois if not p.served]
        self.dist_cache = {}
        if not remaining_pois:
            for u in env.uavs:
                self.routes[u.uid] = []
                self.current_targets[u.uid] = None
                self.paths[u.uid] = []
            return
        best_routes_idx = self._run_ga(env, remaining_pois)
        poi_positions = [(p.y, p.x) for p in remaining_pois]
        self.routes = {}
        self.paths = {}
        for idx, u in enumerate(env.uavs):
            route_idxs = best_routes_idx[idx] if idx < len(best_routes_idx) else []
            coords = [poi_positions[i] for i in route_idxs]
            self.routes[u.uid] = coords
            self.paths[u.uid] = []
        self.current_targets = {}
        for u in env.uavs:
            self._advance_route(u.uid)

    def _advance_route(self, uid: int) -> Optional[Tuple[int, int]]:
        route = self.routes.get(uid, [])
        while route:
            target = route[0]
            poi = self.poi_lookup.get(target)
            if poi is None or poi.served:
                route.pop(0)
                continue
            self.current_targets[uid] = target
            self.paths[uid] = []
            return target
        self.current_targets[uid] = None
        self.paths[uid] = []
        return None

    def _next_action(self, env: MarlEnv, uav: UAV) -> int:
        if getattr(uav, "service_left", 0) > 0 and uav.state == "servicing":
            return 9
        target = self.current_targets.get(uav.uid)
        if target is None:
            return 9
        poi = self.poi_lookup.get(target)
        if poi is None or poi.served:
            route = self.routes.get(uav.uid, [])
            if route:
                route.pop(0)
            self._advance_route(uav.uid)
            target = self.current_targets.get(uav.uid)
            if target is None:
                return 9
        if uav.pos == target:
            return 8
        path = self.paths.get(uav.uid)
        if path is None or not path:
            new_path = self._build_path(env, uav.pos, target)
            self.paths[uav.uid] = new_path
        path = self.paths.get(uav.uid, [])
        if not path:
            return 9
        ny, nx = path.pop(0)
        dy = ny - uav.pos[0]
        dx = nx - uav.pos[1]
        for action, (ady, adx) in ACTIONS.items():
            if (ady, adx) == (dy, dx):
                return action
        return 9

    def _build_path(self, env: MarlEnv, start: Tuple[int, int], target: Tuple[int, int]) -> List[Tuple[int, int]]:
        distmap = self._ensure_distmap(env, target)
        if np.isinf(distmap[start[0], start[1]]):
            return []
        path = greedy_follow_distmap(distmap, start, target)
        return list(path or [])

    def _run_ga(self, env: MarlEnv, pois: Sequence[POI]) -> List[List[int]]:
        n = len(pois)
        num_uavs = max(1, len(env.uavs))
        if n == 0:
            return [[] for _ in range(num_uavs)]
        population = [self.rng.permutation(n).tolist() for _ in range(self.pop_size)]
        def score(gene):
            fitness, _ = self._evaluate_gene(gene, env, pois, num_uavs)
            return fitness
        best_gene = min(population, key=score)
        for _ in range(self.generations):
            scored = [(score(g), g) for g in population]
            scored.sort(key=lambda x: x[0])
            elites = [gene for _, gene in scored[: max(1, self.pop_size // 5)]]
            new_population = [elites[0][:]]
            while len(new_population) < self.pop_size:
                parent1 = self._select_gene(scored)
                parent2 = self._select_gene(scored)
                child = self._crossover(parent1, parent2)
                if self.rng.random() < self.mutation_prob:
                    self._mutate(child)
                new_population.append(child)
            population = new_population
            best_gene = elites[0][:]
        _, best_routes = self._evaluate_gene(best_gene, env, pois, num_uavs)
        while len(best_routes) < len(env.uavs):
            best_routes.append([])
        return best_routes[: len(env.uavs)]

    def _evaluate_gene(
        self,
        gene: Sequence[int],
        env: MarlEnv,
        pois: Sequence[POI],
        num_uavs: int,
    ) -> Tuple[float, List[List[int]]]:
        routes: List[List[int]] = [[] for _ in range(num_uavs)]
        if not gene:
            return 0.0, routes
        for idx, poi_idx in enumerate(gene):
            routes[idx % num_uavs].append(int(poi_idx))
        max_time = 0.0
        base_pos = env.base_xy
        for route in routes:
            current = base_pos
            total = 0.0
            for poi_idx in route:
                target = (pois[poi_idx].y, pois[poi_idx].x)
                total += self._distance_between(current, target, env)
                total += float(getattr(pois[poi_idx], "dwell_ticks_eff", getattr(pois[poi_idx], "dwell_ticks", 1)))
                current = target
            max_time = max(max_time, total)
        return max_time, routes

    def _select_gene(self, scored: Sequence[Tuple[float, List[int]]]) -> List[int]:
        k = min(3, len(scored))
        idx = self.rng.integers(0, len(scored), size=k)
        best = min((scored[i] for i in idx), key=lambda x: x[0])
        return best[1][:]

    def _crossover(self, parent1: Sequence[int], parent2: Sequence[int]) -> List[int]:
        if len(parent1) <= 1:
            return list(parent1)
        a, b = sorted(self.rng.integers(0, len(parent1), size=2))
        if a == b:
            b = min(len(parent1), a + 1)
        child = [None] * len(parent1)
        child[a:b] = parent1[a:b]
        ptr = 0
        for gene in parent2:
            if gene in child:
                continue
            while ptr < len(child) and child[ptr] is not None:
                ptr += 1
            if ptr < len(child):
                child[ptr] = gene
        return [int(g) for g in child if g is not None]

    def _mutate(self, gene: List[int]) -> None:
        if len(gene) < 2:
            return
        a, b = self.rng.integers(0, len(gene), size=2)
        gene[a], gene[b] = gene[b], gene[a]

    def _distance_between(self, source: Tuple[int, int], target: Tuple[int, int], env: MarlEnv) -> float:
        distmap = self._ensure_distmap(env, target)
        value = distmap[source[0], source[1]]
        if np.isinf(value):
            return 1e6
        return float(value)

    def _ensure_distmap(self, env: MarlEnv, target: Tuple[int, int]) -> np.ndarray:
        if target not in self.dist_cache:
            self.dist_cache[target] = bfs_dist(env.grid, target)
        return self.dist_cache[target]

def rollout_policy(
    env: MarlEnv,
    policy: PolicyInterface,
    *,
    log_trajectories: bool = False,
    early_stop_patience: int = 0,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Runs a single episode and returns scalar metrics + per-UAV stats (4.4.x)."""
    observations = env.observations()
    policy.reset()
    done = False
    energy_consumed = 0.0
    trajectories: Dict[int, List[Tuple[int, int]]] = {u.uid: [u.pos] for u in env.uavs} if log_trajectories else {}
    best_served = 0
    last_improve_tick = 0
    early_stop_used = False
    while not done:
        actions = policy.select_actions(env, observations)
        observations, _, done, info = env.step(actions)
        energy_consumed += info.get("energy_spent", 0.0)
        if hasattr(policy, "post_step"):
            try:
                policy.post_step(env)
            except Exception:
                pass
        if log_trajectories:
            for u in env.uavs:
                trajectories[u.uid].append(u.pos)
        served_now = sum(1 for p in env.pois if p.served)
        if served_now > best_served:
            best_served = served_now
            last_improve_tick = env.tick
        if early_stop_patience > 0 and env.tick - last_improve_tick >= early_stop_patience:
            done = True
            early_stop_used = True
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
        "early_stop": early_stop_used,
        "early_stop_patience": early_stop_patience,
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


def evaluate_lurigancho_policy(
    builder_fn: Callable[[int], Tuple[dict, Optional[object], Optional[Dict[str, Callable[[int, int], None]]]]],
    seeds: Sequence[int],
    policy_factory: Callable[[], PolicyInterface],
    *,
    policy_name: str,
    env_mode: str,
    log_trajectories: bool = False,
    early_stop_patience: int = 0,
) -> Tuple[List[Dict[str, float]], Dict[str, float], List[Dict[str, Any]]]:
    rew = RewardWeights()
    per_episode: List[Dict[str, float]] = []
    aggregates: Dict[str, float] = {}
    extra_logs: List[Dict[str, Any]] = []
    for seed in seeds:
        instance, global_obs, hooks = builder_fn(seed)
        env = MarlEnv(instance, rew, global_obs=global_obs, hooks=hooks, env_mode=env_mode, ignore_horizon=False)
        policy = policy_factory()
        metrics, extras = rollout_policy(
            env,
            policy,
            log_trajectories=log_trajectories,
            early_stop_patience=early_stop_patience,
        )
        metrics["phase_id"] = 3
        metrics["seed"] = seed
        per_episode.append(metrics)
        extras.update({"policy": policy_name, "phase_id": 3, "seed": seed})
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


def _log_episode_metrics(policy: str, episodes: Sequence[Dict[str, Any]]) -> None:
    for idx, metrics in enumerate(episodes):
        msg = (
            f"[eval][{policy} ep {idx:03d}] "
            f"seed={metrics.get('seed')} "
            f"cov={float(metrics.get('coverage', 0.0)):.3f} "
            f"dist={float(metrics.get('distance', 0.0)):.1f} "
            f"energy={float(metrics.get('energy_per_uav', 0.0)):.1f} "
            f"rtb={float(metrics.get('rtb', 0.0)):.1f} "
            f"served={float(metrics.get('served', 0.0)):.1f}"
        )
        print(msg)


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def write_eval_episode_json(path: Path, policy_details: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> None:
    payload = []
    for policy, data in policy_details.items():
        episodes = data.get("episodes") or []
        extras = data.get("extras") or []
        merged = []
        for idx, metrics in enumerate(episodes):
            entry = {
                "index": idx,
                "metrics": _sanitize_for_json(metrics),
            }
            if idx < len(extras):
                entry["extras"] = _sanitize_for_json(extras[idx])
            merged.append(entry)
        payload.append({"policy": policy, "episodes": merged})
    write_json(path, {"policies": payload})


def generate_eval_plots(out_dir: Path, policy_details: Dict[str, Dict[str, List[Dict[str, Any]]]], rows: Sequence[Dict[str, float]]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print("[plots] matplotlib not installed; skipping plot generation.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    per_episode_metrics = [
        ("coverage", "Coverage per Episode", "Coverage"),
        ("distance", "Distance per Episode", "Distance (m)"),
        ("energy_per_uav", "Energy per UAV per Episode", "Energy"),
    ]
    for metric, title, ylabel in per_episode_metrics:
        plt.figure()
        plotted = False
        for policy, data in policy_details.items():
            series = [float(ep.get(metric, 0.0)) for ep in data.get("episodes", [])]
            if not series:
                continue
            plt.plot(range(1, len(series) + 1), series, label=policy)
            plotted = True
        if not plotted:
            plt.close()
            continue
        plt.title(title)
        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_per_episode.png")
        plt.close()

    bar_metrics = [
        ("coverage_mean", "Coverage Mean"),
        ("distance_mean", "Distance Mean"),
        ("energy_per_uav_mean", "Energy per UAV Mean"),
        ("rtb_mean", "RTB Mean"),
    ]
    for metric, title in bar_metrics:
        plt.figure()
        policies = []
        values = []
        for row in rows:
            if metric in row:
                policies.append(row["policy"])
                values.append(float(row[metric]))
        if not values:
            plt.close()
            continue
        plt.bar(policies, values, color=["#0072B2", "#E69F00", "#009E73", "#D55E00"][: len(values)])
        plt.title(title)
        plt.ylabel(metric.replace("_", " ").title())
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_bar.png")
        plt.close()


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


def evaluate_lurigancho(args, cfg, env_mode: str) -> None:
    if args.mode != "single":
        raise ValueError("Lurigancho environments currently support --mode single for evaluation.")
    scenario_path = Path(args.scenario_file or "lurigancho_scenario.json")
    map_data = load_lurigancho_map(scenario_path)
    fixed_data = None
    if env_mode == "lurigancho_fixed":
        pois_path = Path(args.pois_file or "lurigancho_pois_val.json")
        fixed_data = load_lurigancho_fixed_data(pois_path)
    seeds = [args.seed + i for i in range(max(1, args.val_count))]
    delta_t_s = float(cfg["simulation_environment"].get("delta_t_s", 1.0))
    max_eval_ticks = int(round((4 * 3600) / max(delta_t_s, 1e-6)))  # ~4 horas simuladas

    def _build_episode(seed: int):
        local_rng = np.random.default_rng(seed)
        if env_mode == "lurigancho_random":
            inst, gobs, hk = build_lurigancho_random_episode(map_data, cfg, local_rng, split="val")
        else:
            inst, gobs, hk = build_lurigancho_fixed_episode(map_data, fixed_data, cfg, local_rng, split="val")
        if "horizon_ticks" in inst:
            inst["horizon_ticks"] = min(int(inst["horizon_ticks"]), max_eval_ticks)
        return inst, gobs, hk

    sample_seed = seeds[0]
    sample_instance, sample_global_obs, sample_hooks = _build_episode(sample_seed)
    rew = RewardWeights()
    env = MarlEnv(sample_instance, rew, global_obs=sample_global_obs, hooks=sample_hooks, env_mode=env_mode, ignore_horizon=False)
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

    builder_fn = lambda seed: _build_episode(seed)

    rows: List[Dict[str, float]] = []
    policy_details: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    marl_eps, marl_stats, marl_logs = evaluate_lurigancho_policy(
        builder_fn,
        seeds,
        lambda: GuidedMarlPolicy(
            agent,
            deterministic=not args.stochastic_eval,
            energy_margin=args.guidance_energy_margin,
            stagnation_steps=args.guidance_stagnation_steps,
            disabled=args.disable_guidance,
        ),
        policy_name="marl",
        env_mode=env_mode,
        log_trajectories=args.log_trajectories,
        early_stop_patience=100,
    )
    print("[eval] MARL aggregate:", marl_stats)
    rows.append({"policy": "marl", **marl_stats})
    policy_details["marl"] = {"episodes": marl_eps, "extras": marl_logs}
    if args.log_per_episode:
        _log_episode_metrics("marl", marl_eps)

    baseline_names: List[str] = []
    if args.baseline:
        baseline_names = [name.strip().lower() for name in args.baseline.split(",") if name.strip()]
    baseline_factories = {
        "greedy": lambda: GreedyBaselinePolicy(),
        "genetic": lambda: GeneticBaselinePolicy(),
    }
    all_logs = list(marl_logs)
    for base_name in baseline_names:
        factory = baseline_factories.get(base_name)
        if factory is None:
            print(f"[eval] baseline '{base_name}' is not supported; skipping.")
            continue
        baseline_eps, baseline_stats, baseline_logs = evaluate_lurigancho_policy(
            builder_fn,
            seeds,
            factory,
            policy_name=base_name,
            env_mode=env_mode,
            log_trajectories=args.log_trajectories,
            early_stop_patience=100,
        )
        print(f"[eval] {base_name} aggregate:", baseline_stats)
        rows.append({"policy": base_name, **baseline_stats})
        policy_details[base_name] = {"episodes": baseline_eps, "extras": baseline_logs}
        if args.log_per_episode:
            _log_episode_metrics(base_name, baseline_eps)
        all_logs.extend(baseline_logs)

    if args.out_csv:
        write_metrics_csv(Path(args.out_csv), rows)
        print(f"[eval] metrics saved to {args.out_csv}")

    if args.episodes_json:
        write_eval_episode_json(Path(args.episodes_json), policy_details)
        print(f"[eval] episode logs saved to {args.episodes_json}")

    if args.plots_dir:
        generate_eval_plots(Path(args.plots_dir), policy_details, rows)

    if args.log_trajectories and all_logs:
        summary = analyze_cooperation(all_logs)
        if args.coop_csv:
            write_metrics_csv(Path(args.coop_csv), summary)
        if args.coop_json:
            write_json(Path(args.coop_json), {"summary": summary, "runs": all_logs})


def evaluate(args) -> None:
    cfg = load_config(args.config)
    env_mode = args.env_mode.lower()
    curriculum = _default_curriculum()
    if env_mode != 'abstract':
        evaluate_lurigancho(args, cfg, env_mode)
        return
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
    ap.add_argument("--env-mode", type=str, default="abstract", choices=["abstract", "lurigancho_random", "lurigancho_fixed"], help="Environment type for evaluation")
    ap.add_argument("--scenario-file", type=str, default=None, help="Path to Lurigancho scenario JSON")
    ap.add_argument("--pois-file", type=str, default=None, help="Path to Lurigancho fixed POIs JSON")
    ap.add_argument("--checkpoint", type=str, default=None, help="Path to actor/critic checkpoint")
    ap.add_argument("--mode", type=str, default="single", choices=["single", "robustness", "sensitivity", "coop"])
    ap.add_argument("--seed", type=int, default=2025, help="Base seed for validation scenario generation")
    ap.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Optional baseline policies (comma-separated, e.g., 'greedy,genetic'; genetic supported in Lurigancho modes)",
    )
    ap.add_argument("--log-trajectories", action="store_true", help="Enable trajectory logging during evaluation (single mode)")
    ap.add_argument("--log-per-episode", action="store_true", help="Print per-episode metrics during evaluation")
    ap.add_argument("--episodes-json", type=str, default=None, help="Path to store detailed per-episode metrics/extras in JSON")
    ap.add_argument("--plots-dir", type=str, default=None, help="Directory to write evaluation plots")
    ap.add_argument("--stochastic-eval", action="store_true", help="Use stochastic action sampling for MARL evaluation instead of deterministic argmax")
    ap.add_argument("--guidance-energy-margin", type=float, default=5.0, help="Margen de energía antes de forzar RTB (0 desactiva)")
    ap.add_argument("--guidance-stagnation-steps", type=int, default=200, help="Pasos sin mejora de coverage antes de RTB forzado (<=0 desactiva)")
    ap.add_argument("--disable-guidance", action="store_true", help="Desactiva la lógica guiada RTB/estancamiento")
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
