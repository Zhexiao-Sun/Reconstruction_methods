"""
Extract camera trajectories from videos using Depth Anything 3 image-only inference.
"""

import glob
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../.."))
DA3_ROOT = os.path.join(PROJECT_ROOT, "models/Depth-Anything-3")
DA3_SRC = os.path.join(DA3_ROOT, "src")
if DA3_SRC not in sys.path:
    sys.path.append(DA3_SRC)

from depth_anything_3.api import DepthAnything3


def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(
            PROJECT_ROOT,
            "pipelines/da3/configs/da3_trajectory_config.yml",
        )
    if not os.path.exists(config_path):
        return {
            "paths": {
                "videos_dir": "dataset/wm_videos/DA3",
                "segments_dir": "dataset/Benchmark",
                "output_base_path": "pipelines/da3/output",
            },
            "model": {
                "model_id": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
                "process_res": 504,
                "process_res_method": "upper_bound_resize",
                "ref_view_strategy": "first",
                "use_ray_pose": False,
                "align_to_input_ext_scale": True,
                "num_frames": 9,
                "frame_stride": None,
                "max_frames": None,
                "apply_scale_factor": True,
            },
            "axis_mapping": {"order": ["x", "z", "y"]},
            "visualization": {"save_plots": True, "dpi": 300},
            "output": {"save_npy": True, "save_csv": True, "save_json": True},
        }
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_model(model_id, device):
    model = DepthAnything3.from_pretrained(model_id)
    model = model.to(device=device)
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
        cv2.imwrite(str(frame_path), frame)
        saved_idx += 1
    cap.release()
    return saved_idx


def run_da3_inference(frames_dir, model, model_cfg):
    frame_paths = sorted(glob.glob(os.path.join(str(frames_dir), "frame_*.jpg")))
    if not frame_paths:
        raise ValueError(f"No frames found in {frames_dir}")

    prediction = model.inference(
        frame_paths,
        process_res=model_cfg.get("process_res", 504),
        process_res_method=model_cfg.get("process_res_method", "upper_bound_resize"),
        ref_view_strategy=model_cfg.get("ref_view_strategy", "first"),
        use_ray_pose=model_cfg.get("use_ray_pose", False),
        align_to_input_ext_scale=model_cfg.get("align_to_input_ext_scale", True),
    )
    return prediction


def _ensure_homogeneous_extrinsics(extrinsics):
    if extrinsics is None:
        raise ValueError("DA3 did not return extrinsics")
    if isinstance(extrinsics, torch.Tensor):
        extrinsics = extrinsics.detach().cpu().numpy()
    if extrinsics.ndim == 4:
        extrinsics = extrinsics[0]
    if extrinsics.shape[-2:] == (3, 4):
        bottom = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=extrinsics.dtype), (extrinsics.shape[0], 1))
        extrinsics = np.concatenate([extrinsics, bottom[:, None, :]], axis=1)
    return extrinsics


def _reorder_positions(positions, axis_order):
    axis_map = {"x": 0, "y": 1, "z": 2}
    if not axis_order or len(axis_order) != 3:
        axis_order = ["x", "z", "y"]
    reordered = np.zeros_like(positions)
    reordered[:, 0] = positions[:, axis_map[axis_order[0]]]
    reordered[:, 1] = positions[:, axis_map[axis_order[1]]]
    reordered[:, 2] = positions[:, axis_map[axis_order[2]]]
    return reordered


def extract_trajectory_from_extrinsics(
    extrinsics,
    scale_factor=None,
    apply_scale_factor=False,
    axis_order=None,
):
    extrinsics = _ensure_homogeneous_extrinsics(extrinsics)
    R_w2c = extrinsics[:, :3, :3]
    t_w2c = extrinsics[:, :3, 3]
    R_c2w = np.transpose(R_w2c, (0, 2, 1))
    t_c2w = -np.matmul(R_c2w, t_w2c[:, :, None])[:, :, 0]

    positions = t_c2w
    if apply_scale_factor and scale_factor is not None:
        positions = positions * float(scale_factor)

    trajectory = _reorder_positions(positions, axis_order)
    orientations = R_c2w
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

    prediction = run_da3_inference(frames_dir, model, config["model"])
    extrinsics = prediction.extrinsics
    scale_factor = getattr(prediction, "scale_factor", None)
    if scale_factor is not None:
        scale_factor = float(np.array(scale_factor).reshape(-1)[0])
    is_metric = bool(getattr(prediction, "is_metric", 0))
    apply_scale = bool(config["model"].get("apply_scale_factor", True))
    scale_applied = apply_scale and scale_factor is not None and not is_metric

    trajectory, orientations, trajectory_2d = extract_trajectory_from_extrinsics(
        extrinsics,
        scale_factor=scale_factor,
        apply_scale_factor=scale_applied,
        axis_order=config.get("axis_mapping", {}).get("order"),
    )

    trajectory_length_meters = float(
        np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)) if len(trajectory) > 1 else 0.0
    )

    trajectory_output_dir = Path(output_dir) / "trajectories"
    segment_output_path = trajectory_output_dir / segment_name
    save_trajectory_data(
        trajectory,
        orientations,
        str(segment_output_path),
        metadata={
            "segment_name": segment_name,
            "scale_factor": scale_factor,
            "is_metric": is_metric,
            "scale_applied": scale_applied,
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
            title=f"DA3 - {segment_name}",
            dpi=config["visualization"].get("dpi", 300),
        )

    info = {
        "segment_name": segment_name,
        "num_frames": len(trajectory),
        "trajectory_length_meters": trajectory_length_meters,
        "scale_factor": scale_factor,
        "is_metric": is_metric,
        "scale_applied": scale_applied,
    }
    return trajectory, orientations, trajectory_2d, info


def process_da3_segments_for_trajectories(videos_dir=None, segments_dir=None, output_dir=None, config_path=None):
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
    process_da3_segments_for_trajectories()


if __name__ == "__main__":
    main()

