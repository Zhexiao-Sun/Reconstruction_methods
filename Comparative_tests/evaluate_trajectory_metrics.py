#!/usr/bin/env python3
"""
Evaluate 2D trajectory accuracy from model JSON outputs.

Metrics:
1) Up-to-scale (all models): Sim(2) Umeyama alignment -> ATE-RMSE + RTA (heading error) + AUC
2) Metric scale (all models): raw waypoint RMSE + RTA + AUC

--no-plots 开关
--plot-segments sample_083_custom_segment_075_377 sample_087_custom_segment_057_358
--plot-models Any4D DA3 MapAnything
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_JSONS = [
    os.path.join(SCRIPT_DIR, "any4d/evaluation_results_any4d_20260117_184404.json"),
    os.path.join(SCRIPT_DIR, "da3/evaluation_results_20260129_185138.json"),
    os.path.join(SCRIPT_DIR, "mapanything/evaluation_results_20260130_181931.json"),
    os.path.join(SCRIPT_DIR, "vggt/evaluation_results_20251113_152448.json"),
    os.path.join(SCRIPT_DIR, "vipe/evaluation_results_20251113_145409.json"),
]


@dataclass
class ModelData:
    name: str
    entries: Dict[str, dict]  # segment_name -> entry


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(SCRIPT_DIR, path))


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
    raise ValueError(
        f"Expected a single model in by_model, got {keys}. "
        "Please clean JSON or pass --model-name."
    )


def _extract_entries(data: dict, model_name: str) -> Dict[str, dict]:
    by_segment = data.get("by_segment", {})
    if by_segment:
        out = {}
        for segment_name, model_map in by_segment.items():
            if model_name in model_map:
                out[segment_name] = model_map[model_name]
        if out:
            return out
    by_model = data.get("by_model", {})
    if model_name in by_model:
        return by_model[model_name]
    raise KeyError(f"Model {model_name} not found in JSON.")


def _as_array(points: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.asarray(list(points), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected (N,2) trajectory, got {arr.shape}.")
    return arr


def _sim2_umeyama(src: np.ndarray, dst: np.ndarray, eps: float = 1e-9) -> Tuple[float, np.ndarray, np.ndarray]:
    """Estimate Sim(2) transform (scale, rotation, translation) from src to dst."""
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


def _rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    if len(pred) != len(gt):
        n = min(len(pred), len(gt))
        pred = pred[:n]
        gt = gt[:n]
    diff = pred - gt
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))


def _heading_errors_deg(pred: np.ndarray, gt: np.ndarray, min_step: float) -> np.ndarray:
    if len(pred) < 2 or len(gt) < 2:
        return np.array([], dtype=np.float64)
    n = min(len(pred), len(gt))
    pred = pred[:n]
    gt = gt[:n]

    dp = pred[1:] - pred[:-1]
    dg = gt[1:] - gt[:-1]
    gt_step = np.linalg.norm(dg, axis=1)
    valid = gt_step >= min_step
    if not np.any(valid):
        return np.array([], dtype=np.float64)

    theta_p = np.arctan2(dp[:, 1], dp[:, 0])
    theta_g = np.arctan2(dg[:, 1], dg[:, 0])
    diff = theta_p - theta_g
    diff = (diff + math.pi) % (2 * math.pi) - math.pi
    err = np.abs(diff[valid])
    return np.degrees(err)


def _auc_from_errors(errors_deg: np.ndarray, max_threshold: float) -> float | None:
    if errors_deg.size == 0:
        return None
    bins = np.arange(int(max_threshold) + 1)
    hist, _ = np.histogram(errors_deg, bins=bins)
    normalized = hist.astype(float) / float(len(errors_deg))
    return float(np.mean(np.cumsum(normalized)))


def _summarize(values: List[float]) -> Dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "std": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
    }


def _collect_segments(models: List[ModelData], mode: str) -> List[str]:
    if mode == "union":
        segs = set()
        for m in models:
            segs.update(m.entries.keys())
        return sorted(segs)
    segs = None
    for m in models:
        curr = set(m.entries.keys())
        segs = curr if segs is None else segs.intersection(curr)
    return sorted(segs or [])


def _evaluate_model(
    model: ModelData,
    segments: List[str],
    mode: str,
    heading_min_step: float,
    auc_thresholds: List[float],
) -> Dict[str, object]:
    per_segment = {}
    ate_values = []
    rta_values = []
    all_heading_errors = []

    for seg in segments:
        entry = model.entries.get(seg)
        if entry is None:
            continue
        pred = _as_array(entry.get("trajectory_2d", []))
        gt = _as_array(entry.get("gt_trajectory_2d", []))

        if mode == "up_to_scale":
            scale, R, t = _sim2_umeyama(pred, gt)
            pred_used = _apply_sim2(pred, scale, R, t)
        else:
            pred_used = pred

        ate_rmse = _rmse(pred_used, gt)
        heading_errors = _heading_errors_deg(pred_used, gt, heading_min_step)
        if heading_errors.size > 0:
            rta_mean = float(np.mean(heading_errors))
            rta_values.append(rta_mean)
            all_heading_errors.append(heading_errors)
        else:
            rta_mean = None

        ate_values.append(ate_rmse)
        per_segment[seg] = {
            "ate_rmse": ate_rmse,
            "rta_mean_deg": rta_mean,
            "num_heading": int(heading_errors.size),
        }

    heading_pool = np.concatenate(all_heading_errors) if all_heading_errors else np.array([])
    auc = {f"auc_{int(t)}deg": _auc_from_errors(heading_pool, t) for t in auc_thresholds}

    return {
        "num_segments": len(per_segment),
        "ate_rmse": _summarize(ate_values),
        "rta_mean_deg": _summarize(rta_values),
        "auc": auc,
        "per_segment": per_segment,
        "num_heading_total": int(heading_pool.size),
    }


def _plot_trajectory(
    segment_name: str,
    model_name: str,
    gt: np.ndarray,
    pred_raw: np.ndarray,
    pred_aligned: np.ndarray,
    output_path: str,
    model_color=None,
    aligned_color=None,
) -> None:
    import matplotlib.pyplot as plt

    if model_color is None:
        model_color = "#0065BD"
    if aligned_color is None:
        aligned_color = "#9467bd" 

    fig, ax = plt.subplots(figsize=(12, 10))

    # Ground truth (match target_eval_mapanything.py style)
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

    # Raw prediction (use model color, with start/end markers)
    ax.plot(
        pred_raw[:, 0],
        pred_raw[:, 1],
        "-",
        color=model_color,
        linewidth=2,
        alpha=0.7,
        label=f"{model_name}",
        zorder=5,
    )
    ax.scatter(pred_raw[0, 0], pred_raw[0, 1], color=model_color, s=80, marker="o", zorder=6)
    ax.scatter(pred_raw[-1, 0], pred_raw[-1, 1], color=model_color, s=80, marker="s", zorder=6)

    # Sim(2)-aligned prediction (distinct color)
    ax.plot(
        pred_aligned[:, 0],
        pred_aligned[:, 1],
        "-",
        color=aligned_color,
        linewidth=2,
        alpha=0.9,
        label=f"{model_name} Sim(2)",
        zorder=7,
    )
    ax.scatter(pred_aligned[0, 0], pred_aligned[0, 1], color=aligned_color, s=80, marker="o", zorder=8)
    ax.scatter(pred_aligned[-1, 0], pred_aligned[-1, 1], color=aligned_color, s=80, marker="s", zorder=8)

    # X limits similar to target_eval_mapanything.py
    all_x_values = gt[:, 0].tolist()
    all_x_values.extend(pred_raw[:, 0].tolist())
    all_x_values.extend(pred_aligned[:, 0].tolist())
    min_x = min(all_x_values)
    max_x = max(all_x_values)
    x_range = max_x - min_x
    if x_range > 0:
        ax.set_xlim(min_x - 0.01 * x_range, max_x + 0.05 * x_range)

    ax.set_xlabel("X (meters)", fontsize=30)
    ax.set_ylabel("Y (meters)", fontsize=30)
    ax.set_title(f"Trajectory Comparison\n{model_name} - {segment_name}", fontsize=36, pad=20)
    ax.tick_params(axis="both", labelsize=26)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=24)
    ax.axis("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    import csv

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 2D trajectory metrics.")
    parser.add_argument("--jsons", nargs="+", default=DEFAULT_JSONS, help="List of JSON files.")
    parser.add_argument("--model-name", nargs="*", default=None, help="Optional model names per JSON.")
    parser.add_argument(
        "--metric-exclude",
        nargs="+",
        default=[],
        help="Model names to exclude from metric-scale evaluation.",
    )
    parser.add_argument(
        "--segment-mode",
        choices=["intersection", "union"],
        default="intersection",
        help="How to collect segments across models.",
    )
    parser.add_argument("--heading-min-step", type=float, default=1e-3, help="Min GT step for heading.")
    parser.add_argument(
        "--auc-thresholds",
        nargs="+",
        type=float,
        default=[5.0, 10.0],
        help="AUC max-degree thresholds.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plotting (by default plots all segments for all models).",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory for results.")
    parser.add_argument(
        "--plot-segments",
        nargs="*",
        default=None,
        help="Segment names to plot (optional). If omitted, plots all evaluated segments.",
    )
    parser.add_argument(
        "--plot-models",
        nargs="*",
        default=None,
        help="Model names to plot (optional). If omitted, plots all models.",
    )
    args = parser.parse_args()

    json_paths = [(_resolve_path(p)) for p in args.jsons]
    model_names = args.model_name or []

    models: List[ModelData] = []
    for idx, json_path in enumerate(json_paths):
        data = _load_json(json_path)
        fallback = model_names[idx] if idx < len(model_names) else None
        model_name = _get_single_model_name(data, fallback=fallback)
        entries = _extract_entries(data, model_name)
        models.append(ModelData(name=model_name, entries=entries))

    output_dir = args.output_dir
    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(SCRIPT_DIR, f"eval_outputs_{ts}")

    up_models = models
    metric_exclude = {m.lower() for m in args.metric_exclude}
    metric_models = [m for m in models if m.name.lower() not in metric_exclude]

    results = {
        "config": {
            "jsons": json_paths,
            "segment_mode": args.segment_mode,
            "heading_min_step": args.heading_min_step,
            "auc_thresholds": args.auc_thresholds,
            "metric_exclude": args.metric_exclude,
        },
        "up_to_scale": {},
        "metric": {},
    }

    for eval_name, eval_models, mode in [
        ("up_to_scale", up_models, "up_to_scale"),
        ("metric", metric_models, "metric"),
    ]:
        segments = _collect_segments(eval_models, args.segment_mode)
        eval_result = {
            "segments": segments,
            "models": {},
        }
        for model in eval_models:
            eval_result["models"][model.name] = _evaluate_model(
                model=model,
                segments=segments,
                mode=mode,
                heading_min_step=args.heading_min_step,
                auc_thresholds=args.auc_thresholds,
            )
        results[eval_name] = eval_result

    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "trajectory_metrics_summary.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    for eval_name in ["up_to_scale", "metric"]:
        rows = []
        for model_name, model_data in results[eval_name]["models"].items():
            for seg, seg_data in model_data["per_segment"].items():
                rows.append(
                    {
                        "eval": eval_name,
                        "model": model_name,
                        "segment": seg,
                        "ate_rmse": seg_data["ate_rmse"],
                        "rta_mean_deg": seg_data["rta_mean_deg"],
                        "num_heading": seg_data["num_heading"],
                    }
                )
        csv_path = os.path.join(output_dir, f"{eval_name}_per_segment.csv")
        _write_csv(
            csv_path,
            rows,
            fieldnames=["eval", "model", "segment", "ate_rmse", "rta_mean_deg", "num_heading"],
        )

    if not args.no_plots:
        import matplotlib.cm as cm

        plot_models = {m.name for m in models}
        if args.plot_models:
            plot_models = {m for m in plot_models if m in set(args.plot_models)}

        models_sorted = sorted(plot_models)
        color_palette = cm.tab20(np.linspace(0, 1, 20))
        model_colors = {name: color_palette[i % 20] for i, name in enumerate(models_sorted)}

        # Default: plot all evaluated segments (up_to_scale includes all models).
        segments_to_plot = args.plot_segments
        if not segments_to_plot:
            segments_to_plot = results["up_to_scale"]["segments"]

        print(f"Plotting {len(segments_to_plot)} segments for {len(plot_models)} models...")

        for model in models:
            if model.name not in plot_models:
                continue
            for seg in segments_to_plot:
                entry = model.entries.get(seg)
                if entry is None:
                    continue
                pred = _as_array(entry.get("trajectory_2d", []))
                gt = _as_array(entry.get("gt_trajectory_2d", []))
                scale, R, t = _sim2_umeyama(pred, gt)
                pred_aligned = _apply_sim2(pred, scale, R, t)
                out_path = os.path.join(output_dir, "plots", f"{model.name}_{seg}.png")
                _plot_trajectory(
                    seg,
                    model.name,
                    gt,
                    pred,
                    pred_aligned,
                    out_path,
                    model_color=model_colors.get(model.name),
                )

    print(f"✓ Results saved to: {results_path}")
    print(f"✓ Per-segment CSVs saved to: {output_dir}")


if __name__ == "__main__":
    main()
