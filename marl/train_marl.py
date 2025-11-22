from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from envgen.base import sample_base_on_perimeter
from envgen.config import load_config
from envgen.energy import energy_dist_map
from envgen.gridsearch import INF, bfs_dist
from envgen.obstacles import generate_obstacles
from envgen.pois import assign_attributes, place_pois
from envgen.sampling import sample_grid_size, sample_n_pois, sample_p_obs
from envgen.sim_engine.entities import POI, UAV
from envgen.sim_engine.utils import ticks_per_cell

from .curriculum import CurriculumManager, CurriculumPhase
from .envs import (build_lurigancho_fixed_episode, build_lurigancho_random_episode, load_lurigancho_fixed_data, load_lurigancho_map)
from .networks import CentralCritic, GraphActor
from .ppo_agent import PPOAgent, Transition
from .spaces import ACTIONS, LocalObservation, build_action_mask, build_global_state_vector, build_local_observation


@dataclass
class RewardWeights:
    poi_reward: float = 10.0  # reward multiplier per priority point served
    timeliness_bonus: float = 1.0  # bonus for meeting TW
    tardiness_penalty: float = 0.1  # penalty per tick served late
    violation_penalty: float = 20.0  # penalty per violated TW
    tick_penalty: float = 0.2  # per-tick penalty to minimize duration
    idle_base_penalty: float = 0.5  # per-idling UAV penalty if POIs pending
    invalid_penalty: float = 0.5  # invalid action penalty
    energy_penalty: float = 0.0  # energy is tracked but not punished
    full_coverage_bonus: float = 60.0  # terminal bonus when all POIs are served
    high_coverage_bonus: float = 30.0  # bonus when coverage exceeds threshold
    high_coverage_threshold: float = 0.9
    incremental_bonus: float = 5.0  # bonus for each incremental step of POIs served
    incremental_step: int = 5
    reward_scale: float = 5.0


def _load_compatible_state_dict(module: torch.nn.Module, state: Dict[str, torch.Tensor], *, module_name: str) -> None:
    """Load checkpoint weights, skipping tensors whose shapes no longer match."""
    current = module.state_dict()
    loadable: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    for key, tensor in state.items():
        if key in current and current[key].shape == tensor.shape:
            loadable[key] = tensor
        else:
            skipped.append(key)
    info = module.load_state_dict(loadable, strict=False)
    if skipped:
        preview = ", ".join(skipped[:5])
        print(f"[ckpt] {module_name}: skipped {len(skipped)} tensors due to shape mismatch ({preview}...).")
    if info.unexpected_keys:
        preview = ", ".join(info.unexpected_keys[:5])
        print(f"[ckpt] {module_name}: unexpected keys ignored ({preview}...).")
    if info.missing_keys:
        preview = ", ".join(info.missing_keys[:5])
        print(f"[ckpt] {module_name}: missing keys left at init ({preview}...).")


def _split_value(cfg_val, split: str) -> float:
    return float(cfg_val[split]) if isinstance(cfg_val, dict) else float(cfg_val)


def make_instance(
    cfg: dict,
    phase: CurriculumPhase,
    rng: np.random.Generator,
    split: str = "train",
    *,
    poi_scale: float = 1.0,
    E_max_scale: float = 1.0,
) -> dict:
    # Sample grid and obstacles
    H, W = sample_grid_size(cfg["simulation_environment"]["grid_size"][split], rng)
    pobs = 0.0 if not phase.obstacles else sample_p_obs(cfg["generation_rules"]["obstacles"]["density_range"], rng)
    pobs *= phase.obstacle_scale
    pobs = float(np.clip(pobs, 0.0, 1.0))
    grid = generate_obstacles(H, W, pobs, rng, clear_perim=True, perim_width=1, min_free_perim_ratio=0.05)

    base_xy = sample_base_on_perimeter(grid, rng)
    distmap = bfs_dist(grid, base_xy)

    # POIs
    ncfg = cfg["generation_rules"]["n_pois"]
    limits = tuple(ncfg["n_pois_size"])
    profiles = ncfg["profiles"]
    n_pois = sample_n_pois(H, W, ncfg[split], limits, profiles, rng)
    n_pois = int(max(1, round(n_pois * phase.poi_multiplier * float(poi_scale))))
    pois_xy = place_pois(grid, n_pois, rng, forbid_xy=base_xy)
    pois_xy = [(y, x) for (y, x) in pois_xy if distmap[y, x] < INF]

    delta_t_s = float(cfg["simulation_environment"]["delta_t_s"])
    horizon_ticks = int(cfg["simulation_environment"]["horizon_ticks"])
    speed_ms = float(cfg["uav_specs"]["cruise_speed_ms"])
    L_o = float(cfg["routes"]["cell_distances_m"]["L_o"])
    L_d = float(cfg["routes"]["cell_distances_m"]["L_d"])
    tps = ticks_per_cell(L_o, speed_ms, delta_t_s)

    def eta_fn(xy: Tuple[int, int]) -> int:
        y, x = xy
        d_cells = int(distmap[y, x])
        if d_cells >= INF:
            return INF
        t_s = (d_cells * L_o) / max(speed_ms, 1e-9)
        return int(np.ceil(t_s / max(delta_t_s, 1e-9)))

    pois_attr = assign_attributes(
        pois_xy,
        cfg_pois=cfg["pois"],
        eta_fn=eta_fn,
        delta_t_s=delta_t_s,
        horizon_ticks=horizon_ticks,
        rng=rng,
    )
    if not phase.time_windows:
        for p in pois_attr:
            p["tw"] = None

    pois: List[POI] = []
    for p in pois_attr:
        poi = POI(
            y=int(p["y"]),
            x=int(p["x"]),
            dwell_ticks=int(p["dwell_ticks"]),
            priority=int(p["priority"]),
            tw=p.get("tw"),
        )
        poi.dwell_ticks_eff = int(p.get("dwell_ticks_eff", poi.dwell_ticks))
        poi.n_persons = int(p.get("n_persons", 0))
        pois.append(poi)

    base_E_max = _split_value(cfg["uav_specs"]["E_max"], split) if phase.energy else 1e9
    E_max = base_E_max * float(E_max_scale)
    E_reserve = _split_value(cfg["uav_specs"]["E_reserve"], split) if phase.energy else 0.0
    e_ortho = float(cfg["uav_specs"]["energy_model"]["e_move_ortho"])
    e_diag_cfg = cfg["uav_specs"]["energy_model"]["e_move_diag"]
    e_diag = _split_value(e_diag_cfg, split)
    e_wait = float(cfg["uav_specs"]["energy_model"].get("e_wait", 0.0))

    energy_map = energy_dist_map(grid, base_xy, e_move_ortho=e_ortho, e_move_diag=e_diag) if phase.energy else np.zeros_like(grid, dtype=float)

    n_uav_cfg = cfg["uav_specs"]["n_uavs"][split]
    n_uavs = int(rng.integers(int(n_uav_cfg[0]), int(n_uav_cfg[1]) + 1))
    uavs = [UAV(uid=i, pos=tuple(base_xy), E=E_max) for i in range(n_uavs)]

    return {
        "grid": grid,
        "base_xy": tuple(base_xy),
        "pois": pois,
        "uavs": uavs,
        "distmap": distmap,
        "energy_map": energy_map,
        "E_max": E_max,
        "E_reserve": E_reserve,
        "e_move_ortho": e_ortho,
        "e_move_diag": e_diag,
        "e_wait": e_wait,
        "horizon_ticks": horizon_ticks,
        "ticks_per_step": tps,
        "L_o": L_o,
        "L_d": L_d,
    }


class MarlEnv:

    def __init__(
        self,
        instance: dict,
        reward_weights: RewardWeights,
        *,
        global_obs: Optional[object] = None,
        hooks: Optional[Dict[str, Callable[[int, int], None]]] = None,
        env_mode: str = "abstract",
        ignore_horizon: bool = False,
    ):

        self.grid = instance["grid"]

        self.base_xy = instance["base_xy"]

        self.pois: List[POI] = instance["pois"]

        self.uavs: List[UAV] = instance["uavs"]

        self.distmap = instance["distmap"]

        self.energy_map = instance["energy_map"]

        self.E_max = instance["E_max"]

        self.E_reserve = instance["E_reserve"]

        self.e_move_ortho = instance["e_move_ortho"]

        self.e_move_diag = instance["e_move_diag"]

        self.e_wait = instance["e_wait"]

        self.horizon_ticks = instance["horizon_ticks"]

        self.ticks_per_step = instance["ticks_per_step"]

        self.L_o = instance.get("L_o", 1.0)

        self.L_d = instance.get("L_d", 1.0)

        self.rewards = reward_weights
        self.env_mode = env_mode
        self.ignore_horizon = ignore_horizon

        self.global_obs = global_obs

        hooks = hooks or {}

        self._on_visit = hooks.get("on_visit")

        self._on_service = hooks.get("on_service")

        self.tick = 0

        self.steps_ortho = {u.uid: 0 for u in self.uavs}

        self.steps_diag = {u.uid: 0 for u in self.uavs}

        self.rtb_count = 0

        self.energy_spent = {u.uid: 0.0 for u in self.uavs}

        self.served_counts = {u.uid: 0 for u in self.uavs}

        # Track action usage per episode for debugging (4.4.3 / action sanity checks)

        self.action_hist: Dict[int, int] = {i: 0 for i in range(len(ACTIONS))}

        if self._on_visit is not None:

            for u in self.uavs:

                self._on_visit(u.pos[0], u.pos[1])



    def _rtb_step(self, uav: UAV) -> float:
        """Moves one step along the gradient to base. Returns energy spent."""
        if uav.pos == self.base_xy:
            uav.state = "idle"
            uav.E = self.E_max
            return 0.0
        y, x = uav.pos
        best = (y, x)
        bestd = self.distmap[y, x]
        energy_spent = 0.0
        diag_step = False
        for a in range(8):
            dy, dx = ACTIONS[a]
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.grid.shape[0] and 0 <= nx < self.grid.shape[1] and not self.grid[ny, nx]:
                if self.distmap[ny, nx] < bestd:
                    best = (ny, nx)
                    bestd = self.distmap[ny, nx]
                    diag = (dy != 0 and dx != 0)
                    energy_spent = self.e_move_diag if diag else self.e_move_ortho
                    diag_step = diag
        uav.pos = best
        uav.E -= energy_spent
        self.energy_spent[uav.uid] += energy_spent
        if diag_step:
            self.steps_diag[uav.uid] += 1
        else:
            self.steps_ortho[uav.uid] += 1
        uav.state = "rtb"
        return energy_spent

    def _apply_move(self, uav: UAV, action: int) -> float:
        dy, dx = ACTIONS[action]
        ny, nx = uav.pos[0] + dy, uav.pos[1] + dx
        diag = (dy != 0 and dx != 0)
        step_cost = self.e_move_diag if diag else self.e_move_ortho
        uav.pos = (ny, nx)
        uav.E -= step_cost
        self.energy_spent[uav.uid] += step_cost
        if self._on_visit is not None:
            self._on_visit(ny, nx)
        if diag:
            self.steps_diag[uav.uid] += 1
        else:
            self.steps_ortho[uav.uid] += 1
        return step_cost

    def _serve_poi(self, uav: UAV, poi: POI) -> Tuple[float, float, int]:
        poi.served = True
        poi.first_visit_t = self.tick
        poi.served_by = uav.uid
        self.served_counts[uav.uid] += 1
        uav.state = "servicing"
        uav.service_left = poi.dwell_ticks_eff
        timeliness_reward = 0.0
        tardiness = 0
        if poi.tw:
            if self.tick > poi.tw["tmax"]:
                poi.violated = True
                tardiness = self.tick - poi.tw["tmax"]
            else:
                timeliness_reward = 1.0
        poi.tardiness = tardiness
        if self._on_service is not None:
            self._on_service(poi.y, poi.x)
        return float(poi.priority), timeliness_reward, tardiness

    def observations(self) -> Dict[int, LocalObservation]:

        obs = {}

        global_vec: Optional[np.ndarray] = None

        if self.global_obs is not None:

            if hasattr(self.global_obs, 'flatten'):

                global_vec = np.asarray(self.global_obs.flatten(), dtype=np.float32)

            else:

                global_vec = np.asarray(self.global_obs, dtype=np.float32)

        for u in self.uavs:

            obs[u.uid] = build_local_observation(

                u,

                self.uavs,

                self.pois,

                self.grid,

                self.base_xy,

                self.distmap,

                self.energy_map,

                tick=self.tick,

                horizon_ticks=self.horizon_ticks,

                E_max=self.E_max,

                E_reserve=self.E_reserve,

                e_move_ortho=self.e_move_ortho,

                e_move_diag=self.e_move_diag,

                e_wait=self.e_wait,

                ticks_per_step=self.ticks_per_step,

                global_feats=global_vec,

            )

        return obs

    def global_state(self) -> np.ndarray:
        return build_global_state_vector(
            self.uavs, self.pois, tick=self.tick, horizon_ticks=self.horizon_ticks, E_max=self.E_max
        )

    def step(self, actions: Dict[int, int]) -> Tuple[Dict[int, LocalObservation], float, bool, dict]:
        served_before = sum(1 for p in self.pois if p.served)
        invalid = 0
        energy_spent = 0.0
        coverage_gain = 0.0
        priority_reward = 0.0
        timeliness_reward = 0.0
        tard_penalty = 0.0
        violations = 0

        poi_lookup = {(p.y, p.x): p for p in self.pois}

        pre_states = {u.uid: u.state for u in self.uavs}
        for u in self.uavs:
            action = actions.get(u.uid, 9)
            mask = build_action_mask(
                u,
                self.grid,
                e_move_ortho=self.e_move_ortho,
                e_move_diag=self.e_move_diag,
                e_wait=self.e_wait,
                E_reserve=self.E_reserve,
                base_xy=self.base_xy,
                pois=self.pois,
            )
            if not mask[action]:
                invalid += 1
                action = 9  # hover fallback
            self.action_hist[action] = self.action_hist.get(action, 0) + 1

            # If already servicing, tick down regardless of action
            if getattr(u, "service_left", 0) > 0 and u.state == "servicing":
                u.service_left -= 1
                energy_spent += self.e_wait
                u.E -= self.e_wait
                self.energy_spent[u.uid] += self.e_wait
                if u.service_left <= 0:
                    u.state = "idle"
                continue

            if action in range(8):
                energy_spent += self._apply_move(u, action)
            elif action == 8:
                poi = poi_lookup.get(u.pos)
                if poi and not poi.served:
                    pr, tr, tard = self._serve_poi(u, poi)
                    priority_reward += pr
                    timeliness_reward += tr
                    tard_penalty += tard
                    if tard > 0:
                        violations += 1
            elif action == 9:
                u.E -= self.e_wait
                energy_spent += self.e_wait
                self.energy_spent[u.uid] += self.e_wait
            elif action == 10:
                energy_spent += self._rtb_step(u)

            u.E = max(u.E, 0.0)
            if pre_states[u.uid] != "rtb" and u.state == "rtb":
                self.rtb_count += 1

        idle_base_count = 0
        if any(not p.served for p in self.pois):
            for u in self.uavs:
                if u.pos == self.base_xy and u.state == "idle" and getattr(u, "service_left", 0) <= 0:
                    idle_base_count += 1

        served_after = sum(1 for p in self.pois if p.served)
        total_pois = max(len(self.pois), 1)
        coverage_ratio = served_after / total_pois
        if served_after > served_before:
            coverage_gain = (served_after - served_before) / total_pois

        incremental_before = served_before // self.rewards.incremental_step
        incremental_after = served_after // self.rewards.incremental_step
        incremental_bonus = (
            (incremental_after - incremental_before) * self.rewards.incremental_bonus
        )
        reward = (
            self.rewards.poi_reward * priority_reward
            + self.rewards.timeliness_bonus * timeliness_reward
            - self.rewards.tardiness_penalty * tard_penalty
            - self.rewards.energy_penalty * energy_spent
            - self.rewards.invalid_penalty * invalid
            - self.rewards.idle_base_penalty * idle_base_count
            - self.rewards.tick_penalty
            - self.rewards.violation_penalty * violations
            + incremental_bonus
        )

        self.tick += 1
        # Episode termination: by default, horizon OR all POIs served.
        # When ignore_horizon is True (used in deterministic evaluation),
        # only full completion of POIs can terminate the episode.
        done = all(p.served for p in self.pois)
        if not self.ignore_horizon:
            done = done or (self.tick >= self.horizon_ticks)
            if self.env_mode == "lurigancho_fixed" and coverage_ratio >= self.rewards.high_coverage_threshold:
                done = True
        if done:
            if coverage_ratio >= 1.0 - 1e-6:
                reward += self.rewards.full_coverage_bonus
            elif coverage_ratio >= self.rewards.high_coverage_threshold:
                reward += self.rewards.high_coverage_bonus
        scale = max(self.rewards.reward_scale, 1e-6)
        reward = reward / scale
        obs = self.observations()
        info = {
            "served": served_after,
            "total_pois": len(self.pois),
            "coverage": coverage_ratio,
            "energy_spent": energy_spent,
            "invalid": invalid,
            "idle_base": idle_base_count,
            "violations": violations,
        }
        return obs, float(reward), bool(done), info


def _default_curriculum() -> CurriculumManager:
    phases = [
        CurriculumPhase(
            name="phase1_simple",
            obstacles=False,
            time_windows=False,
            energy=False,
            poi_multiplier=0.5,
            threshold=0.70,
        ),
        CurriculumPhase(
            name="phase2_intermediate",
            obstacles=True,
            time_windows=True,
            energy=True,
            obstacle_scale=0.6,
            poi_multiplier=0.8,
            threshold=0.78,
        ),
        CurriculumPhase(
            name="phase3_full",
            obstacles=True,
            time_windows=True,
            energy=True,
            obstacle_scale=1.0,
            poi_multiplier=1.0,
            threshold=0.85,
        ),
    ]
    return CurriculumManager(phases, window=5)


def train_loop(args) -> None:
    cfg = load_config(args.config)
    rng = np.random.default_rng(args.seed)
    curriculum = _default_curriculum()
    rew = RewardWeights()

    env_mode = args.env_mode.lower()
    map_data = None
    fixed_data = None
    if env_mode in ("lurigancho_random", "lurigancho_fixed"):
        scenario_path = Path(args.scenario_file or "lurigancho_scenario.json")
        map_data = load_lurigancho_map(scenario_path)
        if env_mode == "lurigancho_fixed":
            pois_path = Path(args.pois_file or "lurigancho_pois_val.json")
            fixed_data = load_lurigancho_fixed_data(pois_path)

    def build_instance(split: str):
        if env_mode == "abstract":
            return make_instance(cfg, curriculum.current, rng, split=split), None, None
        if env_mode == "lurigancho_random":
            return build_lurigancho_random_episode(map_data, cfg, rng, split=split)
        if env_mode == "lurigancho_fixed":
            if fixed_data is None:
                raise ValueError("--pois-file is required for lurigancho_fixed env")
            return build_lurigancho_fixed_episode(map_data, fixed_data, cfg, rng, split=split)
        raise ValueError(f"Unknown env-mode: {env_mode}")

    sample_instance, sample_global_obs, sample_hooks = build_instance("train")
    env = MarlEnv(sample_instance, rew, global_obs=sample_global_obs, hooks=sample_hooks, env_mode=env_mode)
    obs_dim = env.observations()[0].obs_vector.size
    node_dim = env.observations()[0].node_feats.shape[1]
    global_dim = env.global_state().size

    actor = GraphActor(obs_dim=obs_dim, node_feat_dim=node_dim, hidden_dim=128, n_actions=11)
    critic = CentralCritic(state_dim=global_dim, hidden_dim=128)
    agent = PPOAgent(
        actor,
        critic,
        gamma=args.gamma,
        gae_lambda=args.lam,
        actor_lr=args.lr_actor,
        critic_lr=args.lr_critic,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        clip_eps=args.clip_eps,
        adv_clip=args.adv_clip,
        value_clip=args.value_clip,
    )

    if args.pretrained:
        state = torch.load(args.pretrained, map_location=agent.device)
        _load_compatible_state_dict(agent.actor, state["actor"], module_name="actor")
        _load_compatible_state_dict(agent.critic, state["critic"], module_name="critic")
        print(f"[train] loaded pretrained checkpoint {args.pretrained}")

    best_cov = 0.0
    replay_cache: List[Transition] = []

    train_start = time.time()
    logs_dir = Path("results")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"train_log_{env_mode}.csv"
    if not log_path.exists():
        log_path.write_text("episode,coverage,return,adv,phase,loss_pi,loss_v,episode_seconds,total_seconds\n")

    for ep in range(args.episodes):
        ep_start = time.time()
        if ep > 0:
            # refresh instance each episode
            instance, global_obs, hooks = build_instance("train")
            env = MarlEnv(instance, rew, global_obs=global_obs, hooks=hooks, env_mode=env_mode)

        obs = env.observations()
        done = False

        while not done:
            actions: Dict[int, int] = {}
            global_state = env.global_state()
            value = float(agent.critic(torch.tensor(global_state, dtype=torch.float32, device=agent.device).unsqueeze(0)).item())

            for uid, ob in obs.items():
                action, logprob, _, _ = agent.act(
                    ob.obs_vector, ob.node_feats, ob.adj_matrix, ob.action_mask, deterministic=False
                )
                actions[uid] = action
                tr = Transition(
                    obs_vector=ob.obs_vector,
                    node_feats=ob.node_feats,
                    adj=ob.adj_matrix,
                    action_mask=ob.action_mask,
                    action=action,
                    logprob=logprob,
                    reward=0.0,  # filled after step
                    value=value,
                    next_value=0.0,
                    done=False,
                    global_state=global_state,
                )
                agent.store_transition(tr)

            next_obs, reward, done, info = env.step(actions)
            next_value = 0.0
            if not done:
                next_state_vec = env.global_state()
                next_value = float(
                    agent.critic(torch.tensor(next_state_vec, dtype=torch.float32, device=agent.device).unsqueeze(0)).item()
                )

            # Fill rewards/done for the last |uavs| transitions
            for i in range(len(env.uavs)):
                idx = -1 - i
                agent.buffer.data[idx].reward = reward
                agent.buffer.data[idx].done = done
                agent.buffer.data[idx].next_value = next_value

            obs = next_obs

        # Augment buffer with cached experiences (simple replay)
        if replay_cache:
            agent.buffer.data.extend(replay_cache)

        used_data = list(agent.buffer.data)
        if agent.buffer.advantages is None or agent.buffer.returns is None:
            agent.buffer.compute_advantages(agent.gamma, agent.lam)
        mean_ret = float(agent.buffer.returns.mean().item())
        mean_adv = float(agent.buffer.advantages.mean().item())

        stats = agent.update(batch_size=args.batch_size, epochs=args.epochs)
        best_cov = max(best_cov, info.get("coverage", 0.0))
        if env_mode == "abstract":
            advanced = curriculum.update(info.get("coverage", 0.0))
        else:
            advanced = False

        phase_label = curriculum.current.name if env_mode == "abstract" else env_mode
        ep_elapsed = time.time() - ep_start
        total_elapsed = time.time() - train_start
        with log_path.open("a", encoding="utf-8") as lf:
            lf.write(
                f"{ep},{info.get('coverage', 0.0):.5f},{mean_ret:.5f},{mean_adv:.5f},{phase_label},"
                f"{stats['actor_loss']:.6f},{stats['critic_loss']:.6f},{ep_elapsed:.3f},{total_elapsed:.3f}\n"
            )
        print(
            f"[ep {ep:04d}] cov={info.get('coverage', 0.0):.3f} "
            f"R={mean_ret:.3f} "
            f"A={mean_adv:.3f} "
            f"phase={phase_label} "
            f"loss_pi={stats['actor_loss']:.4f} loss_v={stats['critic_loss']:.4f} entr={stats['entropy']:.4f} "
            f"ep_time={ep_elapsed:.1f}s total={total_elapsed:.1f}s"
        )
        if advanced:
            print(f"--> curriculum advanced to {curriculum.current.name}")

        # Retain a slice of past experiences to mitigate forgetting
        retain_ratio = curriculum.current.retain_ratio if env_mode == 'abstract' else curriculum.phases[-1].retain_ratio
        retain_n = int(round(len(used_data) * retain_ratio))
        if retain_n > 0 and used_data:
            idx = rng.choice(len(used_data), size=retain_n, replace=False)
            replay_cache = [used_data[int(i)] for i in idx]
        else:
            replay_cache = []

    if args.save is not None:
        torch.save({"actor": agent.actor.state_dict(), "critic": agent.critic.state_dict()}, args.save)
        print(f"[save] checkpoints stored at {args.save}")
    _plot_training_curves(log_path, env_mode)
    print(f"[train] completed {args.episodes} episodes in {time.time()-train_start:.1f}s")


def _plot_training_curves(log_path: Path, env_mode: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[train] matplotlib not available; skipping curve plots.")
        return

    if not log_path.exists():
        return
    episodes: List[int] = []
    coverages: List[float] = []
    returns: List[float] = []
    actor_losses: List[float] = []
    critic_losses: List[float] = []
    with log_path.open("r", encoding="utf-8") as lf:
        reader = csv.DictReader(lf)
        for row in reader:
            episodes.append(int(row["episode"]))
            coverages.append(float(row["coverage"]))
            returns.append(float(row["return"]))
            actor_losses.append(float(row["loss_pi"]))
            critic_losses.append(float(row["loss_v"]))
    if not episodes:
        return
    plots_dir = Path("results") / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(episodes, coverages, label="coverage")
    plt.title(f"Coverage vs Episode ({env_mode})")
    plt.xlabel("Episode")
    plt.ylabel("Coverage")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / f"{env_mode}_coverage.png")
    plt.close()

    plt.figure()
    plt.plot(episodes, returns, label="return", color="tab:orange")
    plt.title(f"Return vs Episode ({env_mode})")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / f"{env_mode}_return.png")
    plt.close()

    plt.figure()
    plt.plot(episodes, actor_losses, label="actor loss")
    plt.plot(episodes, critic_losses, label="critic loss")
    plt.title(f"Losses vs Episode ({env_mode})")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / f"{env_mode}_loss.png")
    plt.close()


def parse_args(argv=None):
    ap = argparse.ArgumentParser("train_marl")
    ap.add_argument("--config", type=str, required=True, help="Path to config.json")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--save", type=str, default=None, help="Optional path to save model checkpoints")
    ap.add_argument("--env-mode", type=str, default="abstract", choices=["abstract", "lurigancho_random", "lurigancho_fixed"], help="Environment flavor to train on")
    ap.add_argument("--scenario-file", type=str, default=None, help="Path to Lurigancho scenario JSON")
    ap.add_argument("--pois-file", type=str, default=None, help="Path to Lurigancho fixed POIs JSON")
    ap.add_argument("--pretrained", type=str, default=None, help="Checkpoint to load before training")
    ap.add_argument("--lr-actor", type=float, default=3e-4, help="Actor learning rate (4.3.3)")
    ap.add_argument("--lr-critic", type=float, default=3e-4, help="Critic learning rate (4.3.3)")
    ap.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy coefficient (4.3.3)")
    ap.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient (4.3.3)")
    ap.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip epsilon (4.3.3)")
    ap.add_argument("--adv-clip", type=float, default=10.0, help="Absolute clipping for advantages")
    ap.add_argument("--value-clip", type=float, default=0.3, help="Value function clip range (delta)")
    return ap.parse_args(argv)


def main():
    args = parse_args()
    train_loop(args)


if __name__ == "__main__":
    main()
