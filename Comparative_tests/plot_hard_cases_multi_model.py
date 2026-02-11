#!/usr/bin/env python3
"""
Plot multi-model trajectories for the hardest segments.

Given `worst10_by_model.json` produced by `select_worst_segments.py`, this script:
1) Finds the most frequent hard segments across models:
   - Top-K by frequency in `worst_up_to_scale`
   - Top-K by frequency in `worst_metric`
2) For each selected segment, plots:
   - GT trajectory (dashed)
   - Raw decoded trajectory_2d for each of the 5 models

Plot style is aligned with `evaluate_trajectory_metrics.py`.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MODEL_JSONS = [
    os.path.join(SCRIPT_DIR, "any4d/evaluation_results_any4d_20260117_184404.json"),
    os.path.join(SCRIPT_DIR, "da3/evaluation_results_20260129_185138.json"),
    os.path.join(SCRIPT_DIR, "mapanything/evaluation_results_20260130_181931.json"),
    os.path.join(SCRIPT_DIR, "vggt/evaluation_results_20251113_152448.json"),
    os.path.join(SCRIPT_DIR, "vipe/evaluation_results_20251113_145409.json"),
]


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_single_model_name(data: dict, fallback: str | None = None) -> str:
    by_model = data.get("by_model", {})
    keys = [k for k in by_model.keys()]
    if len(keys) == 1:
        return keys[0]
    if fallback:
        return fallback
    raise ValueError(f"Expected a single model in by_model, got {keys}.")


def _extract_entries(data: dict, model_name: str) -> Dict[str, dict]:
    """
    Return mapping segment_name -> entry.
    Prefer by_segment for robust alignment; fallback to by_model.
    """
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
    raise KeyError(f"Model {model_name} not found in JSON.")


def _as_traj(points) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected (N,2) points, got {arr.shape}")
    return arr


def _sim2_umeyama(src: np.ndarray, dst: np.ndarray, eps: float = 1e-9) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate Sim(2) transform (scale, rotation, translation) from src to dst.

    这里直接复用 evaluate_trajectory_metrics.py 中的实现，保证 Up-to-scale (Sim2)
    对齐方式完全一致。
    """
    if len(src) != len(dst):
        n = min(len(src), len(dst))
        src = src[:n]
        dst = dst[:n]
    mean_src = src.mean(axis=0)
    mean_dst = dst.mean(axis=0)
    src_c = src - mean_src
    dst_c = dst - mean_dst

    var_src = np.mean(np.sum(src_c**2, axis=1))
    if var_src < eps:
        scale = 1.0
        R = np.eye(2)
        t = mean_dst - mean_src
        return scale, R, t

    cov = (dst_c.T @ src_c) / len(src)
    U, S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    S_mat = np.eye(2)
    S_mat[-1, -1] = d
    R = U @ S_mat @ Vt
    scale = (S * np.diag(S_mat)).sum() / var_src
    t = mean_dst - scale * (R @ mean_src)
    return float(scale), R, t


def _apply_sim2(src: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return scale * (src @ R.T) + t


def _select_topk_frequent(worst_json: dict, key: str, topk: int) -> List[Tuple[str, int, List[str]]]:
    """
    key: 'worst_up_to_scale' or 'worst_metric'
    Returns list of (segment, freq, models_included).
    """
    cnt = Counter()
    who = defaultdict(list)
    for model, md in worst_json.get("models", {}).items():
        for item in md.get(key, []):
            seg = item.get("segment")
            if not seg:
                continue
            cnt[seg] += 1
            who[seg].append(model)

    items = sorted(cnt.items(), key=lambda x: (-x[1], x[0]))
    top = items[:topk]
    out = []
    for seg, freq in top:
        out.append((seg, freq, sorted(who[seg], key=str.lower)))
    return out


def _plot_multi(
    segment: str,
    gt: np.ndarray,
    pred_by_model: Dict[str, np.ndarray],
    output_path: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))

    # Ground truth (match evaluate_trajectory_metrics.py style)
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
    model_names = sorted(pred_by_model.keys(), key=str.lower)

    for i, m in enumerate(model_names):
        pred = pred_by_model[m]
        color = cmap(i % 10)
        ax.plot(
            pred[:, 0],
            pred[:, 1],
            "-",
            color=color,
            linewidth=2,
            alpha=0.85,
            label=m,
            zorder=5,
        )
        ax.scatter(pred[0, 0], pred[0, 1], color=color, s=80, marker="o", zorder=6)
        ax.scatter(pred[-1, 0], pred[-1, 1], color=color, s=80, marker="s", zorder=6)

    # limits similar to evaluate_trajectory_metrics.py
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

    # 让标题、坐标轴 label 和图之间稍微拉开一点距离，
    # 对齐 plot_ATE_per_segment_cdf.py 中的风格
    ax.set_xlabel("X (meters)", fontsize=30, labelpad=10)
    ax.set_ylabel("Y (meters)", fontsize=30, labelpad=10)
    ax.set_title(f"Hard Case Trajectory Comparison\n{segment}", fontsize=36, pad=20)
    ax.tick_params(axis="both", labelsize=26, pad=6)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=22)
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_multi_sim2(
    segment: str,
    gt: np.ndarray,
    pred_by_model: Dict[str, np.ndarray],
    output_path: str,
) -> None:
    """
    额外画一张 Up-to-scale (Sim2 aligned) 的多模型对比图：
    - 仍然包含同一段轨迹的 GT
    - 但 5 个模型使用 Sim(2) 对齐后的轨迹
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))

    # Ground truth
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
    model_names = sorted(pred_by_model.keys(), key=str.lower)

    aligned_by_model: Dict[str, np.ndarray] = {}
    for m in model_names:
        pred = pred_by_model[m]
        scale, R, t = _sim2_umeyama(pred, gt)
        aligned_by_model[m] = _apply_sim2(pred, scale, R, t)

    for i, m in enumerate(model_names):
        pred_aligned = aligned_by_model[m]
        color = cmap(i % 10)
        ax.plot(
            pred_aligned[:, 0],
            pred_aligned[:, 1],
            "-",
            color=color,
            linewidth=2,
            alpha=0.95,
            label=f"{m} Sim(2)",
            zorder=7,
        )
        ax.scatter(pred_aligned[0, 0], pred_aligned[0, 1], color=color, s=80, marker="o", zorder=8)
        ax.scatter(pred_aligned[-1, 0], pred_aligned[-1, 1], color=color, s=80, marker="s", zorder=8)

    # limits：这里用对齐后的轨迹范围
    all_x = gt[:, 0].tolist()
    all_y = gt[:, 1].tolist()
    for pred_aligned in aligned_by_model.values():
        all_x.extend(pred_aligned[:, 0].tolist())
        all_y.extend(pred_aligned[:, 1].tolist())
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
    ax.set_title(f"Hard Case Trajectory Comparison (Sim2 aligned)\n{segment}", fontsize=36, pad=20)
    ax.tick_params(axis="both", labelsize=26, pad=6)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=22)
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot hardest segments with all models on one figure.")
    parser.add_argument(
        "--worst-json",
        default=os.path.join(SCRIPT_DIR, "eval_outputs_auc3_10/worst10_by_model.json"),
        help="Path to worst10_by_model.json",
    )
    parser.add_argument("--model-jsons", nargs="+", default=DEFAULT_MODEL_JSONS, help="5 model result JSONs.")
    parser.add_argument("--topk", type=int, default=5, help="Top-K frequent hard segments per category.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: alongside worst-json under hard_cases_plots_<ts>/).",
    )
    args = parser.parse_args()

    worst_path = os.path.abspath(args.worst_json)
    worst = _load_json(worst_path)

    up_top = _select_topk_frequent(worst, "worst_up_to_scale", args.topk)
    metric_top = _select_topk_frequent(worst, "worst_metric", args.topk)

    up_segments = [s for s, _, _ in up_top]
    metric_segments = [s for s, _, _ in metric_top]
    segments = up_segments + metric_segments

    # Load model trajectories
    model_entries: Dict[str, Dict[str, dict]] = {}
    for p in args.model_jsons:
        data = _load_json(os.path.abspath(p))
        model_name = _get_single_model_name(data)
        model_entries[model_name] = _extract_entries(data, model_name)

    if args.output_dir:
        out_dir = os.path.abspath(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.dirname(worst_path)
        out_dir = os.path.join(base, f"hard_cases_plots_{ts}")

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "worst_json": worst_path,
        "topk": int(args.topk),
        "up_to_scale_top": [{"segment": s, "freq": f, "models": ms} for s, f, ms in up_top],
        "metric_top": [{"segment": s, "freq": f, "models": ms} for s, f, ms in metric_top],
        "segments_plotted": segments,
        "model_jsons": [os.path.abspath(p) for p in args.model_jsons],
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Plot each segment
    for seg in segments:
        gt = None
        preds: Dict[str, np.ndarray] = {}

        for model_name, entries in model_entries.items():
            entry = entries.get(seg)
            if not entry:
                continue
            pred = _as_traj(entry.get("trajectory_2d", []))
            preds[model_name] = pred
            if gt is None:
                gt = _as_traj(entry.get("gt_trajectory_2d", []))

        if gt is None or not preds:
            continue

        # Basic consistency check (optional): ensure GT matches across models
        for model_name, entries in model_entries.items():
            entry = entries.get(seg)
            if not entry:
                continue
            gt_i = _as_traj(entry.get("gt_trajectory_2d", []))
            if gt_i.shape == gt.shape:
                max_diff = float(np.max(np.abs(gt_i - gt)))
                if max_diff > 1e-6:
                    # still plot using the first GT; differences may come from float formatting
                    pass

        # 一段 sample 生成两张图：
        # 1）raw waypoints vs GT
        # 2）Up-to-scale (Sim2 aligned) vs GT
        out_path_raw = os.path.join(out_dir, f"{seg}.png")
        _plot_multi(seg, gt, preds, out_path_raw)

        out_path_sim2 = os.path.join(out_dir, f"{seg}_sim2.png")
        _plot_multi_sim2(seg, gt, preds, out_path_sim2)

    print(f"✓ Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()

