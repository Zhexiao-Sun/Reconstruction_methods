"""
Extract camera trajectories from videos using MapAnything image-only inference.
"""

import os
import sys
import glob
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../.."))
MAPANYTHING_ROOT = os.path.join(PROJECT_ROOT, "models/map-anything")
sys.path.append(MAPANYTHING_ROOT)

from mapanything.models import MapAnything
from mapanything.utils.image import load_images


def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(
            PROJECT_ROOT,
            "pipelines/mapanything/configs/mapanything_trajectory_config.yml",
        )
    if not os.path.exists(config_path):
        return {
            "paths": {
                "videos_dir": "dataset/wm_videos/mapanything",
                "segments_dir": "dataset/Benchmark",
                "output_base_path": "pipelines/mapanything/output",
            },
            "model": {
                "model_id": "facebook/map-anything",
                "memory_efficient_inference": False,
                "use_amp": True,
                "amp_dtype": "bf16",
                "apply_mask": True,
                "mask_edges": True,
                "apply_confidence_mask": False,
                "confidence_percentile": 10,
                "resize_mode": "fixed_mapping",
                "resolution_set": 518,
                "frame_stride": 1,
                "max_frames": None,
            },
            "visualization": {"save_plots": True, "dpi": 300},
            "output": {"save_npy": True, "save_csv": True, "save_json": True},
        }
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_model(model_id, device):
    model = MapAnything.from_pretrained(model_id).to(device)
    model.eval()
    return model


def extract_video_frames(
    video_path,
    output_dir,
    frame_stride=1,
    max_frames=None,
    num_frames=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return 0

    if num_frames is not None:
        if total_frames <= num_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        if frame_stride is None or frame_stride <= 0:
            frame_stride = 1
        frame_indices = list(range(0, total_frames, frame_stride))
        if max_frames is not None:
            frame_indices = frame_indices[:max_frames]

    saved_idx = 0
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame_path = output_dir / f"frame_{saved_idx:06d}.jpg"
        # OpenCV imwrite expects BGR; keep the original channel order to avoid color shift.
        cv2.imwrite(str(frame_path), frame)
        saved_idx += 1
    cap.release()
    return saved_idx


def run_mapanything_inference(frames_dir, model, model_cfg):
    views = load_images(
        str(frames_dir),
        resize_mode=model_cfg.get("resize_mode", "fixed_mapping"),
        resolution_set=model_cfg.get("resolution_set", 518),
        stride=1,
        verbose=False,
    )
    if not views:
        raise ValueError(f"No frames loaded from {frames_dir}")

    outputs = model.infer(
        views,
        memory_efficient_inference=model_cfg.get("memory_efficient_inference", False),
        use_amp=model_cfg.get("use_amp", True),
        amp_dtype=model_cfg.get("amp_dtype", "bf16"),
        apply_mask=model_cfg.get("apply_mask", True),
        mask_edges=model_cfg.get("mask_edges", True),
        apply_confidence_mask=model_cfg.get("apply_confidence_mask", False),
        confidence_percentile=model_cfg.get("confidence_percentile", 10),
    )
    return outputs


def collect_camera_poses(predictions):
    camera_poses = []
    metric_scaling_factors = []
    for pred in predictions:
        pose = pred.get("camera_poses")
        if pose is None:
            raise ValueError("camera_poses not found in MapAnything outputs")
        if torch.is_tensor(pose):
            pose = pose.detach().cpu().numpy()
        if pose.ndim == 3:
            pose = pose[0]
        camera_poses.append(pose)

        if "metric_scaling_factor" in pred:
            scale = pred["metric_scaling_factor"]
            if torch.is_tensor(scale):
                scale = scale.detach().cpu().numpy()
            metric_scaling_factors.append(float(np.array(scale).reshape(-1)[0]))

    return np.stack(camera_poses, axis=0), metric_scaling_factors


def extract_trajectory_from_camera_poses(camera_poses):
    if isinstance(camera_poses, torch.Tensor):
        camera_poses = camera_poses.detach().cpu().numpy()
    if camera_poses.ndim == 4:
        camera_poses = camera_poses[0]

    positions = camera_poses[:, :3, 3]
    orientations = camera_poses[:, :3, :3]

    trajectory = np.zeros_like(positions)
    trajectory[:, 0] = positions[:, 0]
    trajectory[:, 1] = positions[:, 2]
    trajectory[:, 2] = positions[:, 1]
    trajectory_2d = trajectory[:, :2]

    return trajectory, orientations, trajectory_2d


def load_ground_truth_trajectory(segment_path, segment_name):
    gt_csv_path = os.path.join(segment_path, f"{segment_name}_trajectory.csv")
    if not os.path.exists(gt_csv_path):
        return None
    gt_df = pd.read_csv(gt_csv_path)
    if "x" not in gt_df.columns or "y" not in gt_df.columns:
        return None
    gt_trajectory_2d = gt_df[["x", "y"]].values
    gt_trajectory_2d = gt_trajectory_2d[~np.isnan(gt_trajectory_2d).any(axis=1)]
    return gt_trajectory_2d


def save_trajectory_data(
    trajectory,
    orientations,
    output_path,
    metadata=None,
    save_npy=True,
    save_csv=True,
    save_json=True,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if save_npy:
        np.save(f"{output_path}_trajectory.npy", trajectory)
        np.save(f"{output_path}_orientations.npy", orientations)

    if save_csv:
        trajectory_df = pd.DataFrame(trajectory, columns=["x", "y", "z"])
        trajectory_df["camera_id"] = range(len(trajectory))
        trajectory_df.to_csv(f"{output_path}_trajectory.csv", index=False)

    if save_json:
        payload = {
            "trajectory": trajectory.tolist(),
            "orientations": orientations.tolist(),
            "metadata": metadata or {},
        }
        with open(f"{output_path}_trajectory.json", "w") as f:
            json.dump(payload, f, indent=2)


def visualize_trajectory_with_gt(trajectory, gt_trajectory_2d=None, output_path=None, title="Camera Trajectory", dpi=300):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(trajectory[:, 0], trajectory[:, 1], "b.-", markersize=5, linewidth=2, label="Predicted")
    ax.scatter(trajectory[0, 0], trajectory[0, 1], c="green", s=80, label="Start")
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c="red", s=80, label="End")

    if gt_trajectory_2d is not None:
        ax.plot(gt_trajectory_2d[:, 0], gt_trajectory_2d[:, 1], "r--", linewidth=2, label="Ground Truth")
        ax.scatter(gt_trajectory_2d[0, 0], gt_trajectory_2d[0, 1], c="darkgreen", s=60, label="GT Start")
        ax.scatter(gt_trajectory_2d[-1, 0], gt_trajectory_2d[-1, 1], c="darkred", s=60, label="GT End")

    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title(title)
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def process_single_video(video_path, model, config, output_dir, segments_dir=None):
    segment_name = Path(video_path).stem
    frames_dir = Path(output_dir) / "frames" / segment_name
    desired_num_frames = config["model"].get("num_frames")
    existing_frames = list(frames_dir.glob("frame_*.jpg")) if frames_dir.exists() else []
    if desired_num_frames is not None:
        needs_refresh = len(existing_frames) != int(desired_num_frames)
    else:
        needs_refresh = len(existing_frames) == 0

    if needs_refresh:
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        extract_video_frames(
            video_path,
            frames_dir,
            frame_stride=config["model"].get("frame_stride", 1),
            max_frames=config["model"].get("max_frames"),
            num_frames=desired_num_frames,
        )

    with torch.no_grad():
        outputs = run_mapanything_inference(frames_dir, model, config["model"])

    camera_poses, metric_scaling_factors = collect_camera_poses(outputs)
    trajectory, orientations, trajectory_2d = extract_trajectory_from_camera_poses(camera_poses)
    trajectory_length_meters = float(
        np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)) if len(trajectory) > 1 else 0.0
    )
    metric_scaling_factor = float(np.mean(metric_scaling_factors)) if metric_scaling_factors else None

    trajectory_output_dir = Path(output_dir) / "trajectories"
    segment_output_path = trajectory_output_dir / segment_name
    save_trajectory_data(
        trajectory,
        orientations,
        str(segment_output_path),
        metadata={
            "segment_name": segment_name,
            "metric_scaling_factor": metric_scaling_factor,
            "num_frames": len(trajectory),
        },
        save_npy=config["output"].get("save_npy", True),
        save_csv=config["output"].get("save_csv", True),
        save_json=config["output"].get("save_json", True),
    )

    if config["visualization"].get("save_plots", True):
        gt_trajectory_2d = None
        if segments_dir is not None:
            gt_segment_dir = os.path.join(segments_dir, segment_name)
            gt_trajectory_2d = load_ground_truth_trajectory(gt_segment_dir, segment_name)
        plot_path = Path(output_dir) / f"trajectory_{segment_name}.png"
        visualize_trajectory_with_gt(
            trajectory_2d,
            gt_trajectory_2d=gt_trajectory_2d,
            output_path=str(plot_path),
            title=f"MapAnything - {segment_name}",
            dpi=config["visualization"].get("dpi", 300),
        )

    info = {
        "segment_name": segment_name,
        "num_frames": len(trajectory),
        "trajectory_length_meters": trajectory_length_meters,
        "metric_scaling_factor": metric_scaling_factor,
    }
    return trajectory, orientations, trajectory_2d, info


def process_mapanything_segments_for_trajectories(videos_dir=None, segments_dir=None, output_dir=None, config_path=None):
    config = load_config(config_path)
    if videos_dir is None:
        videos_dir = os.path.join(PROJECT_ROOT, config["paths"]["videos_dir"])
    if segments_dir is None:
        segments_dir = os.path.join(PROJECT_ROOT, config["paths"]["segments_dir"])
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, config["paths"]["output_base_path"])

    os.makedirs(output_dir, exist_ok=True)
    video_files = sorted(glob.glob(os.path.join(videos_dir, "*.mp4")))
    if not video_files:
        print(f"No videos found in {videos_dir}")
        return [], [], [], []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = setup_model(config["model"]["model_id"], device)

    all_trajectories = []
    all_orientations = []
    all_trajectories_2d = []
    segment_info = []

    for video_path in video_files:
        print(f"Processing: {os.path.basename(video_path)}")
        traj, orient, traj_2d, info = process_single_video(
            video_path,
            model,
            config,
            output_dir,
            segments_dir=segments_dir,
        )
        all_trajectories.append(traj)
        all_orientations.append(orient)
        all_trajectories_2d.append(traj_2d)
        segment_info.append(info)

    return all_trajectories, all_orientations, all_trajectories_2d, segment_info


def main():
    process_mapanything_segments_for_trajectories()


if __name__ == "__main__":
    main()

