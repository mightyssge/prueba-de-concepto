# -*- coding: utf-8 -*-
"""
Lee los logs finales de distintos experimentos (run_single_sim_capture)
y genera:
- Una tabla comparativa (CSV + Markdown impreso) con coverage, violaciones, RTB, servidos por UAV.
- Gráficos de barras para coverage, violaciones y RTB.

Uso:
  python compare_runs.py --files results/routes/politic\ with\ obstacles/results.txt results/routes/politic\ without\ obstacles/results.txt results/routes/greedy/results.txt results/routes/genetica/results.txt

Si no se pasan rutas, intenta encontrar "results.txt" en subdirectorios de results/routes/.
"""
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def parse_final_line(line: str):
    clean = line.replace("[final]", "").strip()

    # Captura key=value; cuando el valor es un dict se toma hasta la llave de cierre.
    # Ejemplo: "serv_uav={0: 5, 1: 4}"
    tokens = {}
    for m in re.finditer(r"(\w+)=({[^}]*}|[^\s]+)", clean):
        key, val = m.group(1), m.group(2)
        tokens[key] = val

    def get_str(key: str, default: str = "") -> str:
        return tokens.get(key, default)

    try:
        policy = get_str("policy", "unknown")

        served_str, total_str = "0", "1"
        serv_token = get_str("serv")
        if serv_token and "/" in serv_token:
            served_str, total_str = serv_token.split("/", 1)
        elif serv_token:
            served_str = serv_token
        served = int(served_str)
        total = int(total_str)

        cov_raw = get_str("cov", "0").replace("%", "")
        coverage = float(cov_raw) / 100.0

        violations = int(get_str("viol", "0"))
        rtb = int(get_str("rtb", "0"))

        serv_uav_str = get_str("serv_uav", "{}")
        cov_uav_str = get_str("cov_uav", "{}")
        serv_uav = ast.literal_eval(serv_uav_str) if serv_uav_str else {}
        cov_uav = ast.literal_eval(cov_uav_str) if cov_uav_str else {}

        return {
            "policy": policy,
            "served": served,
            "total": total,
            "coverage": coverage,
            "violations": violations,
            "rtb": rtb,
            "serv_uav": serv_uav,
            "cov_uav": cov_uav,
        }
    except Exception:
        return None


def load_results(files: List[Path]) -> pd.DataFrame:
    rows = []
    for f in files:
        if not f.exists():
            print(f"[warn] no encontrado: {f}")
            continue
        text = f.read_text(encoding="utf-8", errors="ignore").splitlines()
        final_line = None
        for line in reversed(text):
            if "policy=" in line:
                final_line = line
                break
        if not final_line:
            print(f"[warn] sin línea final en {f}")
            continue
        parsed = parse_final_line(final_line)
        if not parsed:
            print(f"[warn] no se pudo parsear {f}: {final_line}")
            continue
        label = f.parent.name
        parsed.update({"label": label, "path": str(f)})
        rows.append(parsed)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["coverage_pct"] = df["coverage"] * 100.0
    return df


def plot_bars(df: pd.DataFrame, col: str, title: str, out: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    df.sort_values(col, inplace=True)
    positions = range(len(df))
    ax.bar(positions, df[col], color="tab:orange")
    ax.set_title(title)
    ax.set_ylabel(col)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(df["label"], rotation=30, ha="right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", type=str, help="Rutas a results.txt; si se omite, se hace glob en results/routes/**/results.txt")
    ap.add_argument("--out-dir", type=str, default="results/routes/compare")
    args = ap.parse_args()

    if args.files:
        files = [Path(p) for p in args.files]
    else:
        preferred = [
            Path("results/routes/poli with obstacles/results.txt"),
            Path("results/routes/poli without obstacles/results.txt"),
            Path("results/routes/greedy2/results.txt"),
            Path("results/routes/genetica2/results.txt"),
        ]
        files = [p for p in preferred if p.exists()]
        if not files:
            files = list(Path("results/routes").rglob("results.txt"))
    df = load_results(files)
    if df.empty:
        print("[error] no se encontraron resultados válidos.")
        return
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tabla
    table_path = out_dir / "comparison.csv"
    df.to_csv(table_path, index=False)
    print("[tabla] guardada en", table_path)
    print(df[["label", "policy", "coverage_pct", "served", "total", "violations", "rtb"]])

    # Plots
    plot_bars(df.copy(), "coverage_pct", "Cobertura (%)", out_dir / "coverage.png")
    plot_bars(df.copy(), "violations", "Violaciones", out_dir / "violations.png")
    plot_bars(df.copy(), "rtb", "RTB events", out_dir / "rtb.png")


if __name__ == "__main__":
    main()
