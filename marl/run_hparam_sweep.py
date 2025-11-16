from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

from envgen.config import load_config

from .eval_marl import (
    MarlActorPolicy,
    build_agent_for_scenarios,
    build_phase_validation_set,
    evaluate_policy,
)
from .train_marl import parse_args as train_parse_args, train_loop, _default_curriculum


DEFAULT_GRID = [
    {"id": "cfg_A", "lr_actor": 3e-4, "lr_critic": 3e-4, "entropy_coef": 0.01, "value_coef": 0.5, "clip_eps": 0.2},
    {"id": "cfg_B", "lr_actor": 5e-4, "lr_critic": 5e-4, "entropy_coef": 0.01, "value_coef": 0.4, "clip_eps": 0.2},
    {"id": "cfg_C", "lr_actor": 3e-4, "lr_critic": 3e-4, "entropy_coef": 0.02, "value_coef": 0.6, "clip_eps": 0.15},
]


def _filter_grid(config_ids: Optional[str]) -> List[Dict[str, float]]:
    grid = list(DEFAULT_GRID)
    if config_ids:
        wanted = {cid.strip() for cid in config_ids.split(",") if cid.strip()}
        grid = [cfg for cfg in grid if cfg["id"] in wanted]
        missing = wanted.difference({cfg["id"] for cfg in grid})
        if missing:
            print(f"[hparam] warning: unknown config ids skipped: {', '.join(sorted(missing))}")
    return grid


def run_hparam_sweep(args) -> None:
    cfg = load_config(args.config)
    curriculum = _default_curriculum()
    phase_id = 3  # fixed Phase-3 validation
    eval_seeds = [args.eval_seed + i for i in range(args.eval_count)]
    rows: List[Dict[str, float]] = []

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    grid = _filter_grid(args.config_ids)
    if not grid:
        raise ValueError("No configurations to run. Provide valid --config-ids or use defaults.")

    for hp in grid:
        print(f"[hparam] starting config {hp['id']}")
        for seed_offset in range(args.seeds_per_config):
            train_seed = args.train_seed + seed_offset
            ckpt_path = checkpoint_dir / f"{hp['id']}_seed{train_seed}.pt"
            print(f"[hparam] config {hp['id']} seed {train_seed} training")
            train_args = [
                "--config",
                args.config,
                "--episodes",
                str(args.train_episodes),
                "--batch-size",
                str(args.train_batch),
                "--epochs",
                str(args.train_epochs),
                "--gamma",
                str(args.gamma),
                "--lam",
                str(args.lam),
                "--seed",
                str(train_seed),
                "--save",
                str(ckpt_path),
                "--lr-actor",
                str(hp["lr_actor"]),
                "--lr-critic",
                str(hp["lr_critic"]),
                "--entropy-coef",
                str(hp["entropy_coef"]),
                "--value-coef",
                str(hp["value_coef"]),
                "--clip-eps",
                str(hp["clip_eps"]),
            ]
            train_loop(train_parse_args(train_args))

            scenarios = build_phase_validation_set(cfg, curriculum, phase_id, eval_seeds)
            agent = build_agent_for_scenarios(scenarios[0].data, str(ckpt_path))
            episodes, stats, extra_logs = evaluate_policy(
                cfg,
                scenarios,
                lambda: MarlActorPolicy(agent),
                policy_name="marl",
                log_trajectories=args.debug_eval,
            )
            if args.debug_eval:
                for idx, (ep_metrics, extra) in enumerate(zip(episodes, extra_logs)):
                    hist = {int(k): int(v) for k, v in (extra.get("action_hist") or {}).items()}
                    print(
                        f"[hparam][eval] cfg={hp['id']} seed={train_seed} ep={idx} "
                        f"phase={int(ep_metrics['phase_id'])} cov={ep_metrics['coverage']:.3f} "
                        f"dist={ep_metrics['distance']:.1f} served={ep_metrics.get('served', 0):.1f}/"
                        f"{ep_metrics.get('total_pois', 0):.1f} rtb={extra.get('rtb_events', 0)} "
                        f"actions={hist}"
                    )
            row = {
                "config_id": hp["id"],
                "seed": train_seed,
                "lr_actor": hp["lr_actor"],
                "lr_critic": hp["lr_critic"],
                "entropy_coef": hp["entropy_coef"],
                "value_coef": hp["value_coef"],
                "clip_eps": hp["clip_eps"],
            }
            row.update(stats)
            rows.append(row)

            if not args.keep_checkpoints and ckpt_path.exists():
                ckpt_path.unlink()

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(rows[0].keys()) if rows else []
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    print(f"[hparam] results -> {out_path}")


def parse_args():
    ap = argparse.ArgumentParser("run_hparam_sweep", description="Utility for 4.3.3 hyperparameter sweeps.")
    ap.add_argument("--config", type=str, required=True, help="Path to config.json")
    ap.add_argument("--train-seed", type=int, default=7000, help="Base training seed")
    ap.add_argument("--seeds-per-config", type=int, default=3, help="# seeds per config")
    ap.add_argument("--train-episodes", type=int, default=40, help="Training episodes per run")
    ap.add_argument("--train-batch", type=int, default=64, help="PPO batch size")
    ap.add_argument("--train-epochs", type=int, default=4, help="PPO epochs per update")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--checkpoint-dir", type=str, default="results/hparam_checkpoints")
    ap.add_argument("--keep-checkpoints", action="store_true", help="Keep intermediate checkpoints")
    ap.add_argument("--eval-seed", type=int, default=9000, help="Base seed for Phase-3 evaluation scenarios")
    ap.add_argument("--eval-count", type=int, default=5, help="# evaluation scenarios")
    ap.add_argument("--out-csv", type=str, default="results/hparam_sweep.csv")
    ap.add_argument("--config-ids", type=str, default=None, help="Comma-separated subset of configs to run (e.g., cfg_A,cfg_B)")
    ap.add_argument("--debug-eval", action="store_true", help="Print per-episode evaluation metrics and action histograms")
    return ap.parse_args()


def main():
    args = parse_args()
    run_hparam_sweep(args)


if __name__ == "__main__":
    main()
