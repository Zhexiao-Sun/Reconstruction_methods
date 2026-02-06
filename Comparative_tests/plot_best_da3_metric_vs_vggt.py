#!/usr/bin/env python3
"""
Plot best (lowest-error) DA3 metric-scale segments, comparing DA3 vs VGGT (GT-scale) vs GT.

This script:
1) Reads <eval-dir>/metric_per_segment.csv
2) Selects the best-K segments for model=DA3 by smallest ate_rmse (metric scale, no alignment)
3) Loads decoded trajectories from:
   - Comparative_tests/da3/evaluation_results_20260129_185138.json
   - Comparative_tests/vggt/evaluation_results_20251113_152448.json
4) For each selected segment, plots GT + DA3 + VGGT on one figure (style aligned with
   evaluate_trajectory_metrics.py) and saves to an output directory.

Outputs:
  - manifest.json with selected segments + DA3 metric scores
  - one PNG per segment
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_EVAL_DIR = os.path.join(SCRIPT_DIR, "eval_outputs_auc3_10")
DEFAULT_DA3_JSON = os.path.join(SCRIPT_DIR, "da3/evaluation_results_20260129_185138.json")
DEFAULT_VGGT_JSON = os.path.join(SCRIPT_DIR, "vggt/evaluation_results_20251113_152448.json")


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_single_model_name(data: dict) -> str:
    keys = list((data.get("by_model") or {}).keys())
    if len(keys) != 1:
        raise ValueError(f"Expected single model key in by_model, got {keys}")
    return keys[0]


def _extract_entries(data: dict, model_name: str) -> Dict[str, dict]:
    by_segment = data.get("by_segment", {})
    if by_segment:
        out: Dict[str, dict] = {}
        for seg, model_map in by_segment.items():
            if model_name in model_map:
                out[seg] = model_map[model_name]
        if out:
            return out
    by_model = data.get("by_model", {})
    if model_name in by_model:
        return by_model[model_name]
    raise KeyError(f"Model {model_name} not found in JSON")


def _as_traj(points) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected (N,2) points, got {arr.shape}")
    return arr


def _read_metric_scores(metric_csv: str, model_name: str) -> List[dict]:
    rows: List[dict] = []
    with open(metric_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"eval", "model", "segment", "ate_rmse", "rta_mean_deg", "num_heading"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns {sorted(missing)} in {metric_csv}")
        for row in reader:
            if row.get("eval") != "metric":
                continue
            if row.get("model") != model_name:
                continue
            ate = row.get("ate_rmse")
            if ate is None or ate.strip() == "" or ate.lower() == "none":
                continue
            rows.append(
                {
                    "segment": row["segment"],
                    "ate_rmse": float(row["ate_rmse"]),
                    "rta_mean_deg": None
                    if not row.get("rta_mean_deg") or row["rta_mean_deg"].lower() == "none"
                    else float(row["rta_mean_deg"]),
                    "num_heading": None
                    if not row.get("num_heading") or row["num_heading"].lower() == "none"
                    else int(float(row["num_heading"])),
                }
            )
    rows.sort(key=lambda r: (r["ate_rmse"], r["segment"]))
    return rows


def _plot_three(
    segment: str,
    gt: np.ndarray,
    da3: np.ndarray,
    vggt: np.ndarray,
    output_path: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))

    # GT (evaluate_trajectory_metrics.py style)
    ax.plot(gt[:, 0], gt[:, 1], "k--", linewidth=3, label="Ground Truth", zorder=10)
    ax.scatter(
        gt[0, 0],
        gt[0, 1],
        color="green",
        s=200,
        marker="o",
        edgecolors="black",
        linewidth=2,
        label="GT Start",
        zorder=11,
    )
    ax.scatter(
        gt[-1, 0],
        gt[-1, 1],
        color="red",
        s=200,
        marker="s",
        edgecolors="black",
        linewidth=2,
        label="GT End",
        zorder=11,
    )

    # DA3 (blue)
    da3_c = "#1f77b4"
    ax.plot(da3[:, 0], da3[:, 1], "-", color=da3_c, linewidth=2.5, alpha=0.9, label="DA3", zorder=6)
    ax.scatter(da3[0, 0], da3[0, 1], color=da3_c, s=90, marker="o", zorder=7)
    ax.scatter(da3[-1, 0], da3[-1, 1], color=da3_c, s=90, marker="s", zorder=7)

    # VGGT† (GT-scale) (gray)
    vggt_c = "#7f7f7f"
    ax.plot(
        vggt[:, 0],
        vggt[:, 1],
        "-",
        color=vggt_c,
        linewidth=2.5,
        alpha=0.95,
        label="VGGT† (GT-scale)",
        zorder=6,
    )
    ax.scatter(vggt[0, 0], vggt[0, 1], color=vggt_c, s=90, marker="o", zorder=7)
    ax.scatter(vggt[-1, 0], vggt[-1, 1], color=vggt_c, s=90, marker="s", zorder=7)

    # Axis limits similar to evaluate_trajectory_metrics.py
    all_x = gt[:, 0].tolist() + da3[:, 0].tolist() + vggt[:, 0].tolist()
    all_y = gt[:, 1].tolist() + da3[:, 1].tolist() + vggt[:, 1].tolist()
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    x_range = max_x - min_x
    y_range = max_y - min_y
    if x_range > 0:
        ax.set_xlim(min_x - 0.05 * x_range, max_x + 0.08 * x_range)
    if y_range > 0:
        ax.set_ylim(min_y - 0.08 * y_range, max_y + 0.08 * y_range)

    ax.set_xlabel("X (meters)", fontsize=30)
    ax.set_ylabel("Y (meters)", fontsize=30)
    ax.set_title(f"Best DA3 (Metric) — Trajectory Comparison\n{segment}", fontsize=34, pad=20)
    ax.tick_params(axis="both", labelsize=26)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=22)
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot best DA3 metric segments vs VGGT† and GT.")
    parser.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR, help="Eval output directory.")
    parser.add_argument("--topk", type=int, default=5, help="Top-K best segments for DA3 (lowest ate_rmse).")
    parser.add_argument("--da3-model-name", default="DA3", help="Model name as in metric_per_segment.csv.")
    parser.add_argument("--da3-json", default=DEFAULT_DA3_JSON, help="DA3 evaluation_results JSON.")
    parser.add_argument("--vggt-json", default=DEFAULT_VGGT_JSON, help="VGGT evaluation_results JSON.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <eval-dir>/best_da3_metric_vs_vggt_<ts>/).",
    )
    args = parser.parse_args()

    eval_dir = os.path.abspath(args.eval_dir)
    metric_csv = os.path.join(eval_dir, "metric_per_segment.csv")
    if not os.path.exists(metric_csv):
        raise FileNotFoundError(f"Missing: {metric_csv}")

    scored = _read_metric_scores(metric_csv, args.da3_model_name)

    da3_data = _load_json(os.path.abspath(args.da3_json))
    vggt_data = _load_json(os.path.abspath(args.vggt_json))
    da3_key = _get_single_model_name(da3_data)
    vggt_key = _get_single_model_name(vggt_data)
    da3_entries = _extract_entries(da3_data, da3_key)
    vggt_entries = _extract_entries(vggt_data, vggt_key)

    selected: List[dict] = []
    for row in scored:
        seg = row["segment"]
        if seg not in da3_entries or seg not in vggt_entries:
            continue
        selected.append(row)
        if len(selected) >= int(args.topk):
            break

    if not selected:
        raise RuntimeError("No segments selected; check DA3/VGGT JSON coverage and metric CSV.")

    if args.output_dir:
        out_dir = os.path.abspath(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(eval_dir, f"best_da3_metric_vs_vggt_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "eval_dir": eval_dir,
        "metric_csv": metric_csv,
        "da3_model_name": args.da3_model_name,
        "topk": int(args.topk),
        "selected": selected,
        "da3_json": os.path.abspath(args.da3_json),
        "vggt_json": os.path.abspath(args.vggt_json),
        "note": "VGGT† uses GT-derived scale (upper-bound reference).",
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("Top segments (best DA3 metric by lowest ate_rmse):")
    for i, row in enumerate(selected, 1):
        print(f"{i:02d}. {row['segment']} | ate_rmse={row['ate_rmse']:.6f} | rta_mean_deg={row['rta_mean_deg']}")

    for row in selected:
        seg = row["segment"]
        da3_traj = _as_traj(da3_entries[seg].get("trajectory_2d", []))
        vggt_traj = _as_traj(vggt_entries[seg].get("trajectory_2d", []))
        gt = _as_traj(da3_entries[seg].get("gt_trajectory_2d", []))
        out_path = os.path.join(out_dir, f"{seg}.png")
        _plot_three(seg, gt, da3_traj, vggt_traj, out_path)

    print(f"✓ Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()

