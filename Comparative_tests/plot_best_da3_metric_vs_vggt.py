#!/usr/bin/env python3
"""
python plot_best_da3_metric_vs_vggt.py \
  --eval-dir eval_outputs_auc3_10 \
  --topk 8 \
  --target-model-name MapAnything

This script:
1) Reads <eval-dir>/metric_per_segment.csv
2) Selects the best-K segments for model=<target> by smallest ate_rmse (metric scale, no alignment)
3) Loads decoded trajectories from 5 model JSONs (Any4D/DA3/MapAnything/VGGT/ViPE)
4) For each selected segment, saves TWO figures (plot style matches plot_hard_cases_multi_model.py):
   - <target> vs GT on one figure
   - all 5 models vs GT on one figure (with <target> plotted first)

Outputs:
  - manifest.json with selected segments + DA3 metric scores
  - two PNGs per segment
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
DEFAULT_ANY4D_JSON = os.path.join(SCRIPT_DIR, "any4d/evaluation_results_any4d_20260117_184404.json")
DEFAULT_MAPANYTHING_JSON = os.path.join(SCRIPT_DIR, "mapanything/evaluation_results_20260130_181931.json")
DEFAULT_VIPE_JSON = os.path.join(SCRIPT_DIR, "vipe/evaluation_results_20251113_145409.json")

DEFAULT_MODEL_JSONS = [
    DEFAULT_ANY4D_JSON,
    DEFAULT_DA3_JSON,
    DEFAULT_MAPANYTHING_JSON,
    DEFAULT_VGGT_JSON,
    DEFAULT_VIPE_JSON,
]

# Default order for multi-model overlay. The chosen target model will be moved to the front.
DEFAULT_PLOT_ORDER = ["DA3", "MapAnything", "ViPE", "Any4D", "VGGT"]


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


def _plot_multi_style(
    segment: str,
    title: str,
    gt: np.ndarray,
    pred_by_model: Dict[str, np.ndarray],
    output_path: str,
    label_map: Dict[str, str] | None = None,
    style_map: Dict[str, dict] | None = None,
    plot_order: List[str] | None = None,
) -> None:
    """
    Plot GT + multiple predicted trajectories on one figure.

    Style is intentionally aligned with `plot_hard_cases_multi_model.py`.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))

    # Ground truth (match plot_hard_cases_multi_model.py style)
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

    cmap = plt.get_cmap("tab10")

    label_map = label_map or {}
    style_map = style_map or {}

    # Stable plot order: preferred first, then the rest
    preferred = plot_order or DEFAULT_PLOT_ORDER
    model_names = [m for m in preferred if m in pred_by_model] + [
        m for m in sorted(pred_by_model.keys(), key=str.lower) if m not in set(preferred)
    ]

    for i, m in enumerate(model_names):
        pred = pred_by_model[m]
        color = cmap(i % 10)
        display = label_map.get(m, m)
        extra_style = style_map.get(m, {})
        ax.plot(
            pred[:, 0],
            pred[:, 1],
            "-",
            color=color,
            linewidth=2,
            alpha=0.85,
            label=display,
            zorder=5,
            **extra_style,
        )
        ax.scatter(pred[0, 0], pred[0, 1], color=color, s=80, marker="o", zorder=6)
        ax.scatter(pred[-1, 0], pred[-1, 1], color=color, s=80, marker="s", zorder=6)

    # limits similar to plot_hard_cases_multi_model.py
    all_x = gt[:, 0].tolist()
    all_y = gt[:, 1].tolist()
    for pred in pred_by_model.values():
        all_x.extend(pred[:, 0].tolist())
        all_y.extend(pred[:, 1].tolist())
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    x_range = max_x - min_x
    y_range = max_y - min_y
    if x_range > 0:
        ax.set_xlim(min_x - 0.05 * x_range, max_x + 0.08 * x_range)
    if y_range > 0:
        ax.set_ylim(min_y - 0.08 * y_range, max_y + 0.08 * y_range)

    ax.set_xlabel("X (meters)", fontsize=30, labelpad=10)
    ax.set_ylabel("Y (meters)", fontsize=30, labelpad=10)
    ax.set_title(f"{title}\n{segment}", fontsize=36, pad=20)
    ax.tick_params(axis="both", labelsize=26, pad=6)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=22)
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot best metric segments for a target model.")
    parser.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR, help="Eval output directory.")
    parser.add_argument("--topk", type=int, default=5, help="Top-K best segments (lowest ate_rmse).")
    parser.add_argument(
        "--target-model-name",
        default="DA3",
        help="Target model name as in metric_per_segment.csv (e.g., DA3, MapAnything).",
    )
    # Backward-compatible alias (deprecated): keep old flag name working
    parser.add_argument(
        "--da3-model-name",
        default=None,
        help="(deprecated) same as --target-model-name",
    )
    parser.add_argument(
        "--model-jsons",
        nargs="+",
        default=DEFAULT_MODEL_JSONS,
        help="5 model result JSONs (Any4D/DA3/MapAnything/VGGT/ViPE).",
    )
    # kept for backward compatibility / convenience
    parser.add_argument("--da3-json", default=DEFAULT_DA3_JSON, help="(unused if --model-jsons is provided)")
    parser.add_argument("--vggt-json", default=DEFAULT_VGGT_JSON, help="(unused if --model-jsons is provided)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <eval-dir>/best_da3_metric_vs_vggt_<ts>/).",
    )
    args = parser.parse_args()

    target_model = args.target_model_name
    if args.da3_model_name:
        target_model = args.da3_model_name

    eval_dir = os.path.abspath(args.eval_dir)
    metric_csv = os.path.join(eval_dir, "metric_per_segment.csv")
    if not os.path.exists(metric_csv):
        raise FileNotFoundError(f"Missing: {metric_csv}")

    scored = _read_metric_scores(metric_csv, target_model)

    # Load all model trajectories
    model_entries: Dict[str, Dict[str, dict]] = {}
    for p in args.model_jsons:
        data = _load_json(os.path.abspath(p))
        model_name = _get_single_model_name(data)
        model_entries[model_name] = _extract_entries(data, model_name)
    if target_model not in model_entries:
        raise KeyError(
            f"Target model '{target_model}' not found in loaded model JSONs: {sorted(model_entries.keys())}"
        )

    selected: List[dict] = []
    for row in scored:
        seg = row["segment"]
        # Require DA3 coverage; for multi-model overlay we also require the others if present.
        if seg not in model_entries[target_model]:
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
        safe_name = "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in target_model]).lower()
        out_dir = os.path.join(eval_dir, f"best_{safe_name}_metric_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "eval_dir": eval_dir,
        "metric_csv": metric_csv,
        "target_model_name": target_model,
        "topk": int(args.topk),
        "selected": selected,
        "model_jsons": [os.path.abspath(p) for p in args.model_jsons],
        "outputs": "two plots per segment: target_vs_GT, all_models_vs_GT",
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Top segments (best {target_model} metric by lowest ate_rmse):")
    for i, row in enumerate(selected, 1):
        print(f"{i:02d}. {row['segment']} | ate_rmse={row['ate_rmse']:.6f} | rta_mean_deg={row['rta_mean_deg']}")

    for row in selected:
        seg = row["segment"]
        # Collect GT from target entry (should match across models for same segment_name)
        target_entry = model_entries[target_model][seg]
        gt = _as_traj(target_entry.get("gt_trajectory_2d", []))

        # Plot 1: target vs GT
        target_traj = _as_traj(target_entry.get("trajectory_2d", []))
        safe_name = "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in target_model]).lower()
        out_path_target = os.path.join(out_dir, f"{seg}_{safe_name}_vs_gt.png")
        _plot_multi_style(
            segment=seg,
            title=f"Best {target_model} (Metric) — {target_model} vs GT",
            gt=gt,
            pred_by_model={target_model: target_traj},
            output_path=out_path_target,
        )

        # Plot 2: all 5 models vs GT
        preds: Dict[str, np.ndarray] = {}
        for model_name, entries in model_entries.items():
            entry = entries.get(seg)
            if not entry:
                continue
            preds[model_name] = _as_traj(entry.get("trajectory_2d", []))

        label_map = {"VGGT": r"VGGT$^\dagger$"}
        style_map = {}
        preferred_order = [target_model] + [m for m in DEFAULT_PLOT_ORDER if m != target_model]

        out_path_all = os.path.join(out_dir, f"{seg}_all_models.png")
        _plot_multi_style(
            segment=seg,
            title=f"Best {target_model} (Metric) — All Models",
            gt=gt,
            pred_by_model=preds,
            output_path=out_path_all,
            label_map=label_map,
            style_map=style_map,
            plot_order=preferred_order,
        )

    print(f"✓ Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()

