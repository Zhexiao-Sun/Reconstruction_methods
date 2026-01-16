#!/usr/bin/env python3
"""
Evaluate World Models Performance using MapAnything (image-only inference).
"""

import os
import sys
import glob
import json
import yaml
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import argparse
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.append(os.path.join(PROJECT_ROOT, "pipelines/mapanything"))
sys.path.append(os.path.join(PROJECT_ROOT, "evaluation"))

from extrinsic_path.extract_extrinsic_path import (
    load_config as load_mapanything_config,
    setup_model,
    extract_video_frames,
    run_mapanything_inference,
    collect_camera_poses,
    extract_trajectory_from_camera_poses,
    load_ground_truth_trajectory,
)
from metrics import compute_all_metrics


def load_evaluation_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "evaluation/config/vggt_config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_world_model_videos(wm_videos_dir=None):
    if wm_videos_dir is None:
        wm_videos_dir = os.path.join(PROJECT_ROOT, "dataset/wm_videos")

    wm_videos = {}
    if not os.path.exists(wm_videos_dir):
        print(f"Warning: World model videos directory not found: {wm_videos_dir}")
        return {}

    model_dirs = [d for d in glob.glob(os.path.join(wm_videos_dir, "*/")) if os.path.isdir(d)]
    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir.rstrip("/"))
        wm_videos[model_name] = {}

        video_files = glob.glob(os.path.join(model_dir, "*.mp4"))
        for video_path in video_files:
            segment_name = os.path.splitext(os.path.basename(video_path))[0]
            wm_videos[model_name][segment_name] = video_path

        print(f"Found {len(wm_videos[model_name])} videos for model: {model_name}")

    return wm_videos


def compute_trajectory_metrics(pred_trajectory_2d, gt_trajectory_2d, eval_config):
    if len(pred_trajectory_2d) == 0 or len(gt_trajectory_2d) == 0:
        return {
            "ade": float("inf"),
            "fde": float("inf"),
            "miss_rate": 100.0,
            "se": 0.0,
            "ac": 0.0,
            "overall_score": 0.0,
            "valid": False,
            "pred_points": len(pred_trajectory_2d),
            "gt_points": len(gt_trajectory_2d),
        }

    metrics_config = eval_config["metrics"]
    overall_config = eval_config["overall_score"]
    ac_params = metrics_config["ac"]
    overall_score_params = {
        "tau_ade": overall_config["tau_ade"],
        "tau_fde": overall_config["tau_fde"],
        "weights": overall_config["weights"],
    }

    metrics = compute_all_metrics(
        pred_trajectory_2d,
        gt_trajectory_2d,
        miss_threshold=metrics_config["miss_threshold"],
        interpolate_mode=metrics_config["interpolate_mode"],
        se_sigma=metrics_config["se_sigma"],
        ac_params=ac_params,
        overall_score_params=overall_score_params,
    )
    metrics["valid"] = True
    return metrics


def evaluate_world_models(
    segments_dir=None,
    wm_videos_dir=None,
    output_dir=None,
    mapanything_config_path=None,
    eval_config_path=None,
    num_samples=None,
):
    if segments_dir is None:
        segments_dir = os.path.join(PROJECT_ROOT, "dataset/Benchmark")
    if wm_videos_dir is None:
        wm_videos_dir = os.path.join(PROJECT_ROOT, "dataset/wm_videos")
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "evaluation_results/mapanything")
    if mapanything_config_path is None:
        mapanything_config_path = os.path.join(
            PROJECT_ROOT, "pipelines/mapanything/configs/mapanything_trajectory_config.yml"
        )
    if eval_config_path is None:
        eval_config_path = os.path.join(PROJECT_ROOT, "evaluation/config/vggt_config.yaml")

    print("=" * 80)
    print("World Models Evaluation using MapAnything (Image-Only)")
    print("=" * 80)

    eval_config = load_evaluation_config(eval_config_path)
    print(f"\nLoaded evaluation config from: {eval_config_path}")
    print(f"  Miss threshold: {eval_config['metrics']['miss_threshold']}m")
    print(
        f"  Overall score weights: ADE={eval_config['overall_score']['weights']['ade']}, "
        f"FDE={eval_config['overall_score']['weights']['fde']}, "
        f"MR={eval_config['overall_score']['weights']['mr']}, "
        f"SE={eval_config['overall_score']['weights']['se']}, "
        f"AC={eval_config['overall_score']['weights']['ac']}"
    )

    map_cfg = load_mapanything_config(mapanything_config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 80)
    print("Loading MapAnything Model (one-time setup)")
    print("=" * 80)
    model = setup_model(map_cfg["model"]["model_id"], device)
    print(f"✓ Model loaded successfully on {device}")

    os.makedirs(output_dir, exist_ok=True)
    segment_dirs = sorted([d for d in glob.glob(os.path.join(segments_dir, "*/")) if os.path.isdir(d)])
    print(f"\nFound {len(segment_dirs)} segments in total")

    if num_samples is not None and num_samples > 0:
        segment_dirs = segment_dirs[:num_samples]
        print(f"Limiting evaluation to {len(segment_dirs)} samples (--num_samples={num_samples})")
    else:
        print(f"Evaluating all {len(segment_dirs)} segments")

    wm_videos = find_world_model_videos(wm_videos_dir)
    print(f"\nFound {len(wm_videos)} world models: {list(wm_videos.keys())}")

    all_results = {"by_model": {}, "by_segment": {}, "summary": {}}

    for model_name, model_videos in wm_videos.items():
        print(f"\n{'=' * 80}")
        print(f"Evaluating Model: {model_name}")
        print(f"{'=' * 80}")

        all_results["by_model"][model_name] = {}

        for segment_dir in tqdm(segment_dirs, desc=f"Processing {model_name}"):
            segment_name = os.path.basename(segment_dir.rstrip("/"))
            if segment_name not in model_videos:
                print(f"Warning: No video found for segment {segment_name} in model {model_name}")
                continue

            wm_video_path = model_videos[segment_name]
            print(f"\nProcessing: {segment_name}")
            print(f"  WM Video: {os.path.basename(wm_video_path)}")

            try:
                frame_dir = os.path.join(output_dir, "video_frames", segment_name)
                if not os.path.exists(frame_dir) or not glob.glob(os.path.join(frame_dir, "frame_*.jpg")):
                    extract_video_frames(
                        wm_video_path,
                        frame_dir,
                        frame_stride=map_cfg["model"].get("frame_stride", 1),
                        max_frames=map_cfg["model"].get("max_frames"),
                        num_frames=map_cfg["model"].get("num_frames"),
                    )  # reuse cached frames if present

                with torch.no_grad():
                    outputs = run_mapanything_inference(frame_dir, model, map_cfg["model"])

                camera_poses, metric_scaling_factors = collect_camera_poses(outputs)
                trajectory, orientations, trajectory_2d = extract_trajectory_from_camera_poses(camera_poses)
                trajectory_length_meters = float(
                    np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)) if len(trajectory) > 1 else 0.0
                )
                metric_scaling_factor = (
                    float(np.mean(metric_scaling_factors)) if metric_scaling_factors else None
                )
            except Exception as e:
                print(f"  ✗ Error processing: {e}")
                import traceback

                traceback.print_exc()
                continue

            gt_trajectory_2d = load_ground_truth_trajectory(segment_dir, segment_name)
            if gt_trajectory_2d is None:
                print(f"  ✗ No ground truth trajectory found")
                continue

            metrics = compute_trajectory_metrics(trajectory_2d, gt_trajectory_2d, eval_config)

            result_entry = {
                "segment_name": segment_name,
                "model_name": model_name,
                "metric_scaling_factor": metric_scaling_factor,
                "trajectory_length": float(trajectory_length_meters),
                "metrics": metrics,
                "trajectory": trajectory.tolist(),
                "trajectory_2d": trajectory_2d.tolist(),
                "gt_trajectory_2d": gt_trajectory_2d.tolist(),
            }

            all_results["by_model"][model_name][segment_name] = result_entry
            if segment_name not in all_results["by_segment"]:
                all_results["by_segment"][segment_name] = {}
            all_results["by_segment"][segment_name][model_name] = result_entry

            print(
                f"  ✓ Overall: {metrics['overall_score']:.4f} | "
                f"ADE: {metrics['ade']:.4f}m, FDE: {metrics['fde']:.4f}m, "
                f"MR: {metrics['miss_rate']:.2f}%, SE: {metrics['se']:.4f}, AC: {metrics['ac']:.4f}"
            )

    print(f"\n{'=' * 80}")
    print("Computing Summary Statistics")
    print(f"{'=' * 80}")

    for model_name, model_results in all_results["by_model"].items():
        if not model_results:
            continue

        ade_values = []
        fde_values = []
        miss_rates = []
        se_values = []
        ac_values = []
        overall_scores = []

        for segment_result in model_results.values():
            metrics = segment_result["metrics"]
            if metrics["valid"]:
                ade_values.append(metrics["ade"])
                fde_values.append(metrics["fde"])
                miss_rates.append(metrics["miss_rate"])
                se_values.append(metrics["se"])
                ac_values.append(metrics["ac"])
                overall_scores.append(metrics["overall_score"])

        summary = {
            "num_segments": len(model_results),
            "num_valid": len(ade_values),
            "overall_score": {
                "mean": float(np.mean(overall_scores)) if overall_scores else None,
                "median": float(np.median(overall_scores)) if overall_scores else None,
                "std": float(np.std(overall_scores)) if overall_scores else None,
                "min": float(np.min(overall_scores)) if overall_scores else None,
                "max": float(np.max(overall_scores)) if overall_scores else None,
            },
            "ade": {
                "mean": float(np.mean(ade_values)) if ade_values else None,
                "median": float(np.median(ade_values)) if ade_values else None,
                "std": float(np.std(ade_values)) if ade_values else None,
                "min": float(np.min(ade_values)) if ade_values else None,
                "max": float(np.max(ade_values)) if ade_values else None,
            },
            "fde": {
                "mean": float(np.mean(fde_values)) if fde_values else None,
                "median": float(np.median(fde_values)) if fde_values else None,
                "std": float(np.std(fde_values)) if fde_values else None,
                "min": float(np.min(fde_values)) if fde_values else None,
                "max": float(np.max(fde_values)) if fde_values else None,
            },
            "miss_rate": {
                "mean": float(np.mean(miss_rates)) if miss_rates else None,
                "median": float(np.median(miss_rates)) if miss_rates else None,
                "std": float(np.std(miss_rates)) if miss_rates else None,
                "min": float(np.min(miss_rates)) if miss_rates else None,
                "max": float(np.max(miss_rates)) if miss_rates else None,
            },
            "se": {
                "mean": float(np.mean(se_values)) if se_values else None,
                "median": float(np.median(se_values)) if se_values else None,
                "std": float(np.std(se_values)) if se_values else None,
                "min": float(np.min(se_values)) if se_values else None,
                "max": float(np.max(se_values)) if se_values else None,
            },
            "ac": {
                "mean": float(np.mean(ac_values)) if ac_values else None,
                "median": float(np.median(ac_values)) if ac_values else None,
                "std": float(np.std(ac_values)) if ac_values else None,
                "min": float(np.min(ac_values)) if ac_values else None,
                "max": float(np.max(ac_values)) if ac_values else None,
            },
        }

        all_results["summary"][model_name] = summary

        print(f"\n{model_name}:")
        print(f"  Segments processed: {summary['num_valid']}/{summary['num_segments']}")
        if summary["overall_score"]["mean"]:
            print(
                f"  Overall Score: {summary['overall_score']['mean']:.4f} ± {summary['overall_score']['std']:.4f}"
            )
            print(f"  ADE: {summary['ade']['mean']:.4f} ± {summary['ade']['std']:.4f} m")
            print(f"  FDE: {summary['fde']['mean']:.4f} ± {summary['fde']['std']:.4f} m")
            print(f"  Miss Rate (2m): {summary['miss_rate']['mean']:.2f} ± {summary['miss_rate']['std']:.2f} %")
            print(f"  SE (Soft Endpoint): {summary['se']['mean']:.4f} ± {summary['se']['std']:.4f}")
            print(f"  AC (Approach Consistency): {summary['ac']['mean']:.4f} ± {summary['ac']['std']:.4f}")

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_json_path = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")

    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    with open(results_json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert_for_json)

    print(f"\n✓ Saved detailed results to: {results_json_path}")
    create_summary_csv(all_results, output_dir, timestamp)
    try:
        create_evaluation_visualizations(all_results, output_dir, timestamp)
    except Exception as e:
        print(f"Warning: Could not create visualizations: {e}")

    return all_results


def create_summary_csv(all_results, output_dir, timestamp):
    segment_model_data = []
    for segment_name, segment_results in all_results["by_segment"].items():
        for model_name, result in segment_results.items():
            metrics = result["metrics"]
            row = {
                "segment": segment_name,
                "model": model_name,
                "metric_scaling_factor": result.get("metric_scaling_factor"),
                "trajectory_length": result["trajectory_length"],
                "overall_score": metrics["overall_score"],
                "ade": metrics["ade"],
                "fde": metrics["fde"],
                "miss_rate": metrics["miss_rate"],
                "se": metrics["se"],
                "ac": metrics["ac"],
                "pred_points": metrics["pred_points"],
                "gt_points": metrics["gt_points"],
            }
            segment_model_data.append(row)

    df_segment_model = pd.DataFrame(segment_model_data)
    csv_path = os.path.join(output_dir, f"segment_model_results_{timestamp}.csv")
    df_segment_model.to_csv(csv_path, index=False)
    print(f"✓ Saved per-segment results to: {csv_path}")

    summary_data = []
    for model_name, summary in all_results["summary"].items():
        row = {
            "model": model_name,
            "num_segments": summary["num_segments"],
            "num_valid": summary["num_valid"],
            "overall_score_mean": summary["overall_score"]["mean"],
            "overall_score_std": summary["overall_score"]["std"],
            "ade_mean": summary["ade"]["mean"],
            "ade_std": summary["ade"]["std"],
            "fde_mean": summary["fde"]["mean"],
            "fde_std": summary["fde"]["std"],
            "miss_rate_mean": summary["miss_rate"]["mean"],
            "miss_rate_std": summary["miss_rate"]["std"],
            "se_mean": summary["se"]["mean"],
            "se_std": summary["se"]["std"],
            "ac_mean": summary["ac"]["mean"],
            "ac_std": summary["ac"]["std"],
        }
        summary_data.append(row)

    df_summary = pd.DataFrame(summary_data)
    csv_path = os.path.join(output_dir, f"model_summary_{timestamp}.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"✓ Saved model summary to: {csv_path}")


def create_evaluation_visualizations(all_results, output_dir, timestamp):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    models = list(all_results["summary"].keys())
    models_sorted = sorted(models, key=lambda x: (x != "gt_video", x))
    color_palette = cm.tab20(np.linspace(0, 1, 20))
    model_colors = {name: color_palette[i % 20] for i, name in enumerate(models_sorted)}

    fig, axes = plt.subplots(2, 3, figsize=(22, 16))
    metrics_to_plot = ["overall_score", "ade", "fde", "miss_rate", "se", "ac"]
    titles = ["Overall Score", "ADE (m)", "FDE (m)", "Miss Rate (%)", "SE", "AC"]

    for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        ax = axes[idx // 3, idx % 3]
        valid_models = [m for m in models_sorted if all_results["summary"][m][metric]["mean"] is not None]
        means = [all_results["summary"][m][metric]["mean"] for m in valid_models]
        if means:
            x_pos = np.arange(len(valid_models))
            colors = [model_colors[m] for m in valid_models]
            ax.bar(x_pos, means, alpha=0.7, color=colors)
            ax.set_xticks(x_pos + 0.4)
            ax.set_xticklabels(valid_models, rotation=75, ha="right", fontsize=30)
            ax.set_title(title, fontweight="bold" if metric == "overall_score" else "normal", fontsize=36)
            ax.tick_params(axis="y", labelsize=26)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15)
    plt.savefig(os.path.join(output_dir, f"model_comparison_{timestamp}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    for segment_name, segment_results in all_results["by_segment"].items():
        fig, ax = plt.subplots(figsize=(12, 10))
        first_result = list(segment_results.values())[0]
        gt_traj = np.array(first_result["gt_trajectory_2d"])
        ax.plot(gt_traj[:, 0], gt_traj[:, 1], "k--", linewidth=3, label="Ground Truth", zorder=10)
        ax.scatter(gt_traj[0, 0], gt_traj[0, 1], c="green", s=200, marker="o", edgecolors="black", linewidth=2, label="GT Start", zorder=11)
        ax.scatter(gt_traj[-1, 0], gt_traj[-1, 1], c="red", s=200, marker="s", edgecolors="black", linewidth=2, label="GT End", zorder=11)

        all_x_values = gt_traj[:, 0].tolist()
        for model_name, result in segment_results.items():
            pred_traj = np.array(result["trajectory_2d"])
            color = model_colors[model_name]
            ax.plot(pred_traj[:, 0], pred_traj[:, 1], "-", color=color, linewidth=2, alpha=0.7, label=f"{model_name}")
            ax.scatter(pred_traj[0, 0], pred_traj[0, 1], c=color, s=80, marker="o", zorder=5)
            ax.scatter(pred_traj[-1, 0], pred_traj[-1, 1], c=color, s=80, marker="s", zorder=5)
            all_x_values.extend(pred_traj[:, 0].tolist())

        min_x = min(all_x_values)
        max_x = max(all_x_values)
        x_range = max_x - min_x
        ax.set_xlim(min_x - 0.01 * x_range, max_x + 0.05 * x_range)
        ax.set_xlabel("X (meters)", fontsize=30)
        ax.set_ylabel("Y (meters)", fontsize=30)
        ax.set_title(f"Trajectory Comparison\nMapAnything - {segment_name}", fontsize=36, pad=20)
        ax.tick_params(axis="both", labelsize=26)
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=24)
        ax.axis("equal")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"trajectory_{segment_name}_{timestamp}.png"), dpi=300, bbox_inches="tight")
        plt.close()

    print(f"✓ Saved {len(all_results['by_segment'])} trajectory comparison plots")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate World Models Performance using MapAnything (Image-Only)"
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all samples)",
    )
    parser.add_argument("--segments_dir", type=str, default=None)
    parser.add_argument("--wm_videos_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--eval_config_path", type=str, default=None)

    args = parser.parse_args()

    results = evaluate_world_models(
        segments_dir=args.segments_dir,
        wm_videos_dir=args.wm_videos_dir,
        output_dir=args.output_dir,
        mapanything_config_path=args.config_path,
        eval_config_path=args.eval_config_path,
        num_samples=args.num_samples,
    )

    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)
    output_dir_to_show = args.output_dir if args.output_dir else os.path.join(PROJECT_ROOT, "evaluation_results/mapanything")
    print(f"Results saved to: {output_dir_to_show}")
    print("\nTo view results:")
    print(f"  - Summary CSV: {output_dir_to_show}/model_summary_*.csv")
    print(f"  - Detailed results: {output_dir_to_show}/segment_model_results_*.csv")
    print(f"  - Full JSON: {output_dir_to_show}/evaluation_results_*.json")
    print(f"  - Visualizations: {output_dir_to_show}/*.png")


if __name__ == "__main__":
    main()

