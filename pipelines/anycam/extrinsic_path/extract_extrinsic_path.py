"""
Extract camera trajectories from AnyCam output for Target-Bench videos.
"""

from __future__ import annotations

import json
import os
import sys
import glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANYCAM_ROOT = PROJECT_ROOT / "models" / "anycam"
if str(ANYCAM_ROOT) not in sys.path:
    sys.path.append(str(ANYCAM_ROOT))

from anycam.loss import make_loss
from anycam.scripts.common import get_checkpoint_path, load_model
from anycam.scripts.fit_video import fit_video
from anycam.utils.geometry import se3_ensure_numerical_accuracy


def load_anycam_pipeline_config(config_path: str | Path | None = None):
    default_config = {
        "model": {
            "model_path": str(PROJECT_ROOT / "models" / "anycam" / "pretrained_models" / "anycam_seq8"),
            "checkpoint_path": None,
            "device": "cuda",
            "ba_refinement": True,
            "fit_video_config": str(PROJECT_ROOT / "models" / "anycam" / "anycam" / "configs" / "eval_cfgs" / "fit_video.yaml"),
        },
        "data": {
            "segments_dir": str(PROJECT_ROOT / "dataset" / "Benchmark"),
            "videos_dir": str(PROJECT_ROOT / "dataset" / "wm_videos" / "anycam"),
            "output_dir": str(PROJECT_ROOT / "pipelines" / "anycam" / "output"),
            "image_size": 336,
            "target_fps": 0,
        },
        "trajectory": {
            "axis_order": [0, 2, 1],
            "use_scale_factor": False,
            "scale_factor": 1.0,
        },
    }

    if config_path and Path(config_path).exists():
        config = OmegaConf.load(config_path)
        config = OmegaConf.merge(default_config, config)
    else:
        config = OmegaConf.create(default_config)

    return config


def load_anycam_model(model_path: Path, checkpoint_path: Path | None, device: str):
    model_path = Path(model_path)
    if model_path.is_dir():
        config_path = model_path
    else:
        config_path = model_path.parent

    training_config = OmegaConf.load(config_path / "training_config.yaml")
    training_config["model"]["use_provided_flow"] = False
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else get_checkpoint_path(model_path)

    model = load_model(training_config, checkpoint_path).to(device)
    model.eval()

    criterion = None
    if training_config.get("loss"):
        try:
            criterion = [make_loss(cfg) for cfg in training_config.get("loss", [])][0]
        except Exception:
            criterion = None

    return model, criterion


def resize_keep_aspect(frame: np.ndarray, target_size: int | None):
    if target_size is None or target_size <= 0:
        return frame
    height, width = frame.shape[:2]
    if height < width:
        new_height = target_size
        new_width = int((target_size / height) * width)
    else:
        new_width = target_size
        new_height = int((target_size / width) * height)
    return cv2.resize(frame, (new_width, new_height))


def load_video_frames(video_path: Path, target_fps: int, image_size: int | None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    stride = 1
    if target_fps > 0 and fps > 0:
        stride = max(1, round(fps / target_fps))

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = resize_keep_aspect(frame, image_size)
        frames.append(frame.astype(np.float32) / 255.0)
        frame_idx += 1

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames extracted from video: {video_path}")

    return frames, fps


def build_fit_video_config(config_path: Path | None, ba_refinement: bool, image_size: int | None):
    if config_path and Path(config_path).exists():
        config = OmegaConf.load(config_path)
    else:
        config = OmegaConf.create({})

    config["do_ba_refinement"] = bool(ba_refinement)
    dataset_config = config.get("dataset", {})
    dataset_config["image_size"] = [image_size, None]
    config["dataset"] = dataset_config

    return config


def normalize_anycam_poses(trajectory):
    if isinstance(trajectory, torch.Tensor):
        poses = trajectory.detach().cpu().numpy()
    else:
        poses = np.stack([np.array(pose) for pose in trajectory], axis=0)
    return poses


def extract_trajectory_from_c2w(poses: np.ndarray, axis_order: list[int], scale_factor: float):
    translations = poses[:, :3, 3]
    translations = translations[:, axis_order] * scale_factor  # axis_order aligns to top-down plane
    perm = np.eye(3)[axis_order]
    rotations = poses[:, :3, :3]
    rotations = perm @ rotations @ perm.T
    trajectory_2d = translations[:, :2]
    return translations, rotations, trajectory_2d


def run_anycam_on_video(
    video_path: Path,
    model,
    criterion,
    fit_video_config,
    axis_order,
    scale_factor,
    target_fps,
    image_size,
):
    frames, _ = load_video_frames(video_path, target_fps, image_size)

    trajectory, proj, extras_dict, _ = fit_video(
        fit_video_config,
        model,
        criterion,
        frames,
        return_extras=True,
    )

    if isinstance(trajectory, list):
        trajectory = [se3_ensure_numerical_accuracy(torch.tensor(pose)).cpu().numpy() for pose in trajectory]
    poses = normalize_anycam_poses(trajectory)

    if scale_factor is None:
        scale_factor = float(extras_dict.get("scale_factor", 1.0))

    trajectory_3d, orientations, trajectory_2d = extract_trajectory_from_c2w(
        poses,
        axis_order=axis_order,
        scale_factor=scale_factor,
    )

    return trajectory_3d, orientations, trajectory_2d, proj, extras_dict


def load_ground_truth_trajectory(segment_path, segment_name):
    try:
        gt_csv_path = os.path.join(segment_path, f"{segment_name}_trajectory.csv")
        if not os.path.exists(gt_csv_path):
            print(f"Warning: Ground truth trajectory file not found: {gt_csv_path}")
            return None

        gt_df = pd.read_csv(gt_csv_path)
        if "x" not in gt_df.columns or "y" not in gt_df.columns:
            print(f"Warning: GT trajectory CSV missing x/y columns: {gt_csv_path}")
            return None

        gt_trajectory_2d = gt_df[["x", "y"]].values
        gt_trajectory_2d = gt_trajectory_2d[~np.isnan(gt_trajectory_2d).any(axis=1)]
        print(f"✓ Loaded GT trajectory with {len(gt_trajectory_2d)} points from {os.path.basename(gt_csv_path)}")
        return gt_trajectory_2d
    except Exception as e:
        print(f"Error loading GT trajectory from {gt_csv_path}: {e}")
        return None


def save_trajectory_data(trajectory, orientations, output_path, frame_info=None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(f"{output_path}_trajectory.npy", trajectory)
    np.save(f"{output_path}_orientations.npy", orientations)

    trajectory_df = pd.DataFrame(trajectory, columns=["x", "y", "z"])
    trajectory_df["frame_id"] = range(len(trajectory))
    if frame_info:
        for key, value in frame_info.items():
            if isinstance(value, (int, float, str, bool)):
                trajectory_df[key] = value
    trajectory_df.to_csv(f"{output_path}_trajectory.csv", index=False)

    trajectory_data = {
        "trajectory": trajectory.tolist(),
        "orientations": orientations.tolist(),
        "metadata": frame_info if frame_info else {},
    }
    with open(f"{output_path}_trajectory.json", "w") as f:
        json.dump(trajectory_data, f, indent=2)

    print(f"✓ Saved trajectory data to {output_path}_trajectory.*")


def visualize_trajectory_with_gt(trajectory, gt_trajectory_2d=None, output_path=None, title="Camera Trajectory"):
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(16, 8))

        ax1 = fig.add_subplot(131, projection="3d")
        ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], "b.-", markersize=4, linewidth=2)
        ax1.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], c="green", s=100, label="Start")
        ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], c="red", s=100, label="End")
        ax1.set_xlabel("X (meters)")
        ax1.set_ylabel("Y (meters)")
        ax1.set_zlabel("Z (meters)")
        ax1.set_title(f"3D View - {title}")
        ax1.legend()

        ax2 = fig.add_subplot(132)
        ax2.plot(trajectory[:, 0], trajectory[:, 1], "b.-", markersize=6, linewidth=2, label="AnyCam")
        ax2.scatter(trajectory[0, 0], trajectory[0, 1], c="green", s=100, label="Start")
        ax2.scatter(trajectory[-1, 0], trajectory[-1, 1], c="red", s=100, label="End")

        if gt_trajectory_2d is not None:
            ax2.plot(gt_trajectory_2d[:, 0], gt_trajectory_2d[:, 1], "r.-", markersize=4, linewidth=2, alpha=0.7, label="Ground Truth")
            ax2.scatter(gt_trajectory_2d[0, 0], gt_trajectory_2d[0, 1], c="darkgreen", s=80, marker="s", label="GT Start")
            ax2.scatter(gt_trajectory_2d[-1, 0], gt_trajectory_2d[-1, 1], c="darkred", s=80, marker="s", label="GT End")

        ax2.set_xlabel("X (meters)")
        ax2.set_ylabel("Y (meters)")
        ax2.set_title(f"Top-Down View - {title}")
        ax2.legend()
        ax2.axis("equal")
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(133)
        if gt_trajectory_2d is not None:
            pred_2d = trajectory[:, :2]
            ax3.plot(pred_2d[:, 0], pred_2d[:, 1], "b-", linewidth=3, alpha=0.8, label="AnyCam")
            ax3.plot(gt_trajectory_2d[:, 0], gt_trajectory_2d[:, 1], "r--", linewidth=2, alpha=0.8, label="Ground Truth")
            ax3.scatter(pred_2d[0, 0], pred_2d[0, 1], c="blue", s=100, marker="o")
            ax3.scatter(pred_2d[-1, 0], pred_2d[-1, 1], c="blue", s=100, marker="s")
            ax3.scatter(gt_trajectory_2d[0, 0], gt_trajectory_2d[0, 1], c="red", s=80, marker="o")
            ax3.scatter(gt_trajectory_2d[-1, 0], gt_trajectory_2d[-1, 1], c="red", s=80, marker="s")
            ax3.set_title("Trajectory Comparison")
        else:
            pred_2d = trajectory[:, :2]
            ax3.plot(pred_2d[:, 0], pred_2d[:, 1], "b.-", markersize=6, linewidth=2)
            ax3.scatter(pred_2d[0, 0], pred_2d[0, 1], c="green", s=100)
            ax3.scatter(pred_2d[-1, 0], pred_2d[-1, 1], c="red", s=100)
            ax3.set_title("AnyCam Trajectory")

        ax3.set_xlabel("X (meters)")
        ax3.set_ylabel("Y (meters)")
        ax3.legend()
        ax3.axis("equal")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(f"{output_path}_trajectory_plot.png", dpi=300, bbox_inches="tight")
            print(f"✓ Saved trajectory plot to {output_path}_trajectory_plot.png")

        plt.show()
    except ImportError:
        print("Matplotlib not available for visualization")


def process_anycam_segments_for_trajectories(config_path: str | Path | None = None):
    config = load_anycam_pipeline_config(config_path)

    model_cfg = config.model
    data_cfg = config.data
    traj_cfg = config.trajectory

    model_path = Path(model_cfg.model_path)
    checkpoint_path = model_cfg.checkpoint_path
    device = model_cfg.device
    ba_refinement = bool(model_cfg.ba_refinement)
    fit_video_config_path = Path(model_cfg.fit_video_config) if model_cfg.fit_video_config else None

    output_dir = Path(data_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    axis_order = list(traj_cfg.axis_order)
    scale_factor = float(traj_cfg.scale_factor) if traj_cfg.use_scale_factor else None

    model, criterion = load_anycam_model(model_path, checkpoint_path, device)
    fit_video_config = build_fit_video_config(fit_video_config_path, ba_refinement, data_cfg.image_size)

    video_files = sorted(glob.glob(os.path.join(data_cfg.videos_dir, "*.mp4")) + glob.glob(os.path.join(data_cfg.videos_dir, "*.MP4")))
    if not video_files:
        print(f"No video files found in {data_cfg.videos_dir}")
        return [], [], [], []

    all_trajectories = []
    all_orientations = []
    all_trajectories_2d = []
    segment_info = []

    for video_path in video_files:
        segment_name = os.path.basename(video_path).replace(".mp4", "").replace(".MP4", "")
        print(f"\nProcessing segment: {segment_name}")

        try:
            trajectory, orientations, trajectory_2d, proj, extras = run_anycam_on_video(
                Path(video_path),
                model,
                criterion,
                fit_video_config,
                axis_order,
                scale_factor,
                data_cfg.target_fps,
                data_cfg.image_size,
            )

            all_trajectories.append(trajectory)
            all_orientations.append(orientations)
            all_trajectories_2d.append(trajectory_2d)

            trajectory_length = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
            info = {
                "segment_name": segment_name,
                "num_frames": len(trajectory),
                "trajectory_length_meters": float(trajectory_length),
            }
            segment_info.append(info)

            output_path = output_dir / f"{segment_name}"
            save_trajectory_data(trajectory, orientations, str(output_path), info)
        except Exception as e:
            print(f"Error processing {segment_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return all_trajectories, all_orientations, all_trajectories_2d, segment_info


def process_single_segment_demo(segment_name: str, config_path: str | Path | None = None):
    config = load_anycam_pipeline_config(config_path)
    model_cfg = config.model
    data_cfg = config.data
    traj_cfg = config.trajectory

    model_path = Path(model_cfg.model_path)
    checkpoint_path = model_cfg.checkpoint_path
    device = model_cfg.device
    ba_refinement = bool(model_cfg.ba_refinement)
    fit_video_config_path = Path(model_cfg.fit_video_config) if model_cfg.fit_video_config else None

    axis_order = list(traj_cfg.axis_order)
    scale_factor = float(traj_cfg.scale_factor) if traj_cfg.use_scale_factor else None

    model, criterion = load_anycam_model(model_path, checkpoint_path, device)
    fit_video_config = build_fit_video_config(fit_video_config_path, ba_refinement, data_cfg.image_size)

    video_path = Path(data_cfg.videos_dir) / f"{segment_name}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    trajectory, orientations, trajectory_2d, proj, extras = run_anycam_on_video(
        video_path,
        model,
        criterion,
        fit_video_config,
        axis_order,
        scale_factor,
        data_cfg.target_fps,
        data_cfg.image_size,
    )

    segment_path = Path(data_cfg.segments_dir) / segment_name
    gt_trajectory_2d = load_ground_truth_trajectory(segment_path, segment_name)

    output_dir = Path(data_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{segment_name}"
    save_trajectory_data(trajectory, orientations, str(output_path), {"segment_name": segment_name})
    visualize_trajectory_with_gt(trajectory, gt_trajectory_2d, str(output_path), f"AnyCam: {segment_name}")


def main():
    print("=== Extracting Trajectories from AnyCam ===")
    process_anycam_segments_for_trajectories()


if __name__ == "__main__":
    main()

