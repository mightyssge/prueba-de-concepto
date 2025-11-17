from __future__ import annotations

import argparse
from pathlib import Path

from .train_marl import parse_args as train_parse_args, train_loop
from .eval_marl import parse_args as eval_parse_args, evaluate


def _run_train(arg_list):
    args = train_parse_args(arg_list)
    train_loop(args)


def _run_eval(arg_list):
    args = eval_parse_args(arg_list)
    evaluate(args)


def parse_args():
    ap = argparse.ArgumentParser("run_lurigancho_pipeline")
    ap.add_argument("--config", type=str, required=True, help="Path to config.json")
    ap.add_argument("--scenario-file", type=str, default="lurigancho_scenario.json")
    ap.add_argument("--pois-file", type=str, default="lurigancho_pois_val.json")
    ap.add_argument("--seed", type=int, default=1400)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--step1-episodes", type=int, default=200)
    ap.add_argument("--step2-episodes", type=int, default=300)
    ap.add_argument("--step3-episodes", type=int, default=200)
    ap.add_argument("--step1-checkpoint", type=str, default="results/ckpt_step1.pt")
    ap.add_argument("--step2-checkpoint", type=str, default="results/ckpt_step2_lurigancho_random.pt")
    ap.add_argument("--step3-checkpoint", type=str, default="results/ckpt_step3_lurigancho_fixed.pt")
    ap.add_argument("--eval-count", type=int, default=10)
    ap.add_argument("--eval-out", type=str, default="results/lurigancho_fixed_eval.csv")
    ap.add_argument("--baseline", type=str, default="greedy")
    ap.add_argument("--skip-eval", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    Path(args.step1_checkpoint).parent.mkdir(parents=True, exist_ok=True)
    Path(args.step2_checkpoint).parent.mkdir(parents=True, exist_ok=True)
    Path(args.step3_checkpoint).parent.mkdir(parents=True, exist_ok=True)
    # Step 1: abstract pre-training
    if args.step1_episodes > 0:
        train_args = [
            "--config",
            args.config,
            "--episodes",
            str(args.step1_episodes),
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--seed",
            str(args.seed),
            "--env-mode",
            "abstract",
            "--save",
            args.step1_checkpoint,
        ]
        _run_train(train_args)
    elif not Path(args.step1_checkpoint).exists():
        raise FileNotFoundError(f"Step 1 checkpoint not found: {args.step1_checkpoint}")
    # Step 2: Lurigancho random fine-tuning
    if args.step2_episodes > 0:
        train_args = [
            "--config",
            args.config,
            "--episodes",
            str(args.step2_episodes),
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--seed",
            str(args.seed + 1),
            "--env-mode",
            "lurigancho_random",
            "--scenario-file",
            args.scenario_file,
            "--pretrained",
            args.step1_checkpoint,
            "--save",
            args.step2_checkpoint,
        ]
        _run_train(train_args)
    elif not Path(args.step2_checkpoint).exists():
        raise FileNotFoundError(f"Step 2 checkpoint not found: {args.step2_checkpoint}")
    # Step 3: Lurigancho fixed specialization (optional)
    if args.step3_episodes > 0:
        train_args = [
            "--config",
            args.config,
            "--episodes",
            str(args.step3_episodes),
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--seed",
            str(args.seed + 2),
            "--env-mode",
            "lurigancho_fixed",
            "--scenario-file",
            args.scenario_file,
            "--pois-file",
            args.pois_file,
            "--pretrained",
            args.step2_checkpoint,
            "--save",
            args.step3_checkpoint,
        ]
        _run_train(train_args)
        final_checkpoint = args.step3_checkpoint
    else:
        final_checkpoint = args.step2_checkpoint

    if args.skip_eval:
        return

    eval_args = [
        "--config",
        args.config,
        "--env-mode",
        "lurigancho_fixed",
        "--scenario-file",
        args.scenario_file,
        "--pois-file",
        args.pois_file,
        "--checkpoint",
        final_checkpoint,
        "--mode",
        "single",
        "--val-phase",
        "3",
        "--val-count",
        str(args.eval_count),
        "--baseline",
        args.baseline,
        "--out-csv",
        args.eval_out,
    ]
    _run_eval(eval_args)


if __name__ == "__main__":
    main()
