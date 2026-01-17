"""
Extract camera trajectories from videos using Any4D image-only inference.
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
ANY4D_ROOT = os.path.join(PROJECT_ROOT, "models/Any4D")
if ANY4D_ROOT not in sys.path:
    sys.path.append(ANY4D_ROOT)

from any4d.models import init_model
from any4d.utils.hf_utils.hf_helpers import init_hydra_config
from any4d.utils.inference import loss_of_one_batch_multi_view, postprocess_model_outputs_for_inference
from any4d.utils.image import load_images
from any4d.utils.geometry import quaternion_to_rotation_matrix


def _resolve_path(path_value):
    if path_value is None:
        return None
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(PROJECT_ROOT, path_value)


def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(
            PROJECT_ROOT,
            "pipelines/any4d/configs/any4d_trajectory_config.yml",
        )
    if not os.path.exists(config_path):
        return {
            "paths": {
                "videos_dir": "dataset/wm_videos/Any4D",
                "segments_dir": "dataset/Benchmark",
                "output_base_path": "pipelines/any4d/output",
                "any4d_root": "models/Any4D",
            },
            "model": {
                "config_path": "models/Any4D/configs/train.yaml",
                "checkpoint_path": "models/Any4D/checkpoints/any4d_4v_combined.pth",
                "config_overrides": [
                    "machine=local",
                    "model=any4d",
                    "model.encoder.uses_torch_hub=false",
                    "model/task=images_only",
                ],
                "data_norm_type": "dinov2",
                "resize_mode": "fixed_mapping",
                "resolution_set": 518,
                "fixed_size": [518, 336],
                "patch_size": 14,
                "use_amp": True,
                "amp_dtype": "bf16",
                "apply_mask": False,
                "mask_edges": False,
                "apply_confidence_mask": False,
                "confidence_percentile": 10,
                "compute_moge_mask": False,
                "binary_mask_path": None,
                "num_frames": None,
                "frame_stride": 1,
                "max_frames": None,
            },
            "axis_mapping": {"order": ["x", "z", "y"]},
            "visualization": {"save_plots": True, "dpi": 300},
            "output": {"save_npy": True, "save_csv": True, "save_json": True},
        }
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_model(model_cfg, device):
    config_path = _resolve_path(model_cfg.get("config_path"))
    overrides = model_cfg.get("config_overrides")
    if config_path is None or not os.path.exists(config_path):
        raise FileNotFoundError("Any4D config not found; check model.config_path")
    cfg = init_hydra_config(config_path, overrides=overrides)
    model = init_model(cfg.model.model_str, cfg.model.model_config)
    checkpoint_path = _resolve_path(model_cfg.get("checkpoint_path"))
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
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


def _load_any4d_views(frames_dir, model_cfg, device):
    resize_mode = model_cfg.get("resize_mode", "fixed_mapping")
    size = model_cfg.get("fixed_size")
    if size is not None:
        size = tuple(size)
    compute_moge_mask = model_cfg.get("compute_moge_mask", False)
    moge_model = None
    if compute_moge_mask:
        from any4d.utils.moge_inference import load_moge_model

        moge_root = os.path.join(ANY4D_ROOT, "any4d/external/MoGe")
        moge_model = load_moge_model(model_code_path=moge_root, device=device)

    return load_images(
        str(frames_dir),
        resize_mode=resize_mode,
        size=size,
        norm_type=model_cfg.get("data_norm_type", "dinov2"),
        patch_size=model_cfg.get("patch_size", 14),
        resolution_set=model_cfg.get("resolution_set", 518),
        stride=1,
        compute_moge_mask=compute_moge_mask,
        moge_model=moge_model,
        binary_mask_path=model_cfg.get("binary_mask_path"),
    )


def run_any4d_inference(frames_dir, model, model_cfg, device):
    views = _load_any4d_views(frames_dir, model_cfg, device)
    with torch.no_grad():
        result = loss_of_one_batch_multi_view(
            views,
            model,
            None,
            device,
            use_amp=model_cfg.get("use_amp", True),
            amp_dtype=model_cfg.get("amp_dtype", "bf16"),
        )
    raw_outputs = [result[f"pred{i + 1}"] for i in range(len(views))]
    processed_outputs = postprocess_model_outputs_for_inference(
        raw_outputs,
        views,
        apply_mask=model_cfg.get("apply_mask", False),
        mask_edges=model_cfg.get("mask_edges", False),
        apply_confidence_mask=model_cfg.get("apply_confidence_mask", False),
        confidence_percentile=model_cfg.get("confidence_percentile", 10),
    )
    return processed_outputs


def _build_camera_pose_from_quat_trans(cam_quats, cam_trans):
    if torch.is_tensor(cam_quats):
        cam_quats = cam_quats.detach().cpu()
    if torch.is_tensor(cam_trans):
        cam_trans = cam_trans.detach().cpu()
    if cam_quats.ndim == 1:
        cam_quats = cam_quats.unsqueeze(0)
        cam_trans = cam_trans.unsqueeze(0)
    rotation_matrices = quaternion_to_rotation_matrix(cam_quats)
    batch_size = rotation_matrices.shape[0]
    pose = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    pose[:, :3, :3] = rotation_matrices
    pose[:, :3, 3] = cam_trans
    return pose


def collect_camera_poses(predictions):
    camera_poses = []
    metric_scaling_factors = []
    for pred in predictions:
        pose = pred.get("camera_poses")
        if pose is None:
            cam_trans = pred.get("cam_trans")
            cam_quats = pred.get("cam_quats")
            if cam_trans is None or cam_quats is None:
                raise ValueError("Any4D outputs do not contain camera poses")
            pose = _build_camera_pose_from_quat_trans(cam_quats, cam_trans)
        if torch.is_tensor(pose):
            pose = pose.detach().cpu().numpy()
        if pose.ndim == 3 and pose.shape[0] == 1:
            pose = pose[0]
        camera_poses.append(pose)

        if "metric_scaling_factor" in pred:
            scale = pred["metric_scaling_factor"]
            if torch.is_tensor(scale):
                scale = scale.detach().cpu().numpy()
            metric_scaling_factors.append(float(np.array(scale).reshape(-1)[0]))

    return np.stack(camera_poses, axis=0), metric_scaling_factors


def _apply_axis_mapping(positions, axis_order):
    axis_map = {"x": 0, "y": 1, "z": 2}
    indices = [axis_map[a] for a in axis_order]
    return positions[:, indices]


def extract_trajectory_from_camera_poses(camera_poses, axis_order=None):
    if isinstance(camera_poses, torch.Tensor):
        camera_poses = camera_poses.detach().cpu().numpy()
    if camera_poses.ndim == 4:
        camera_poses = camera_poses[0]
    positions = camera_poses[:, :3, 3]
    orientations = camera_poses[:, :3, :3]
    axis_order = axis_order or ["x", "z", "y"]
    mapped_positions = _apply_axis_mapping(positions, axis_order)
    trajectory = np.zeros_like(mapped_positions)
    trajectory[:, 0] = mapped_positions[:, 0]
    trajectory[:, 1] = mapped_positions[:, 1]
    trajectory[:, 2] = mapped_positions[:, 2]
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
    model_cfg = config.get("model", {})
    axis_order = config.get("axis_mapping", {}).get("order", ["x", "z", "y"])
    frames_dir = Path(output_dir) / "frames" / segment_name
    desired_num_frames = model_cfg.get("num_frames")
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
            frame_stride=model_cfg.get("frame_stride", 1),
            max_frames=model_cfg.get("max_frames"),
            num_frames=desired_num_frames,
        )

    device = next(model.parameters()).device
    outputs = run_any4d_inference(frames_dir, model, model_cfg, device)
    camera_poses, metric_scaling_factors = collect_camera_poses(outputs)
    trajectory, orientations, trajectory_2d = extract_trajectory_from_camera_poses(camera_poses, axis_order=axis_order)
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
        save_npy=config.get("output", {}).get("save_npy", True),
        save_csv=config.get("output", {}).get("save_csv", True),
        save_json=config.get("output", {}).get("save_json", True),
    )

    if config.get("visualization", {}).get("save_plots", True):
        gt_trajectory_2d = None
        if segments_dir is not None:
            gt_segment_dir = os.path.join(segments_dir, segment_name)
            gt_trajectory_2d = load_ground_truth_trajectory(gt_segment_dir, segment_name)
        plot_path = Path(output_dir) / f"trajectory_{segment_name}.png"
        visualize_trajectory_with_gt(
            trajectory_2d,
            gt_trajectory_2d=gt_trajectory_2d,
            output_path=str(plot_path),
            title=f"Any4D - {segment_name}",
            dpi=config.get("visualization", {}).get("dpi", 300),
        )

    info = {
        "segment_name": segment_name,
        "num_frames": len(trajectory),
        "trajectory_length_meters": trajectory_length_meters,
        "metric_scaling_factor": metric_scaling_factor,
    }
    return trajectory, orientations, trajectory_2d, info


def process_any4d_segments_for_trajectories(videos_dir=None, segments_dir=None, output_dir=None, config_path=None):
    config = load_config(config_path)
    paths_cfg = config.get("paths", {})
    if videos_dir is None:
        videos_dir = _resolve_path(paths_cfg.get("videos_dir"))
    if segments_dir is None:
        segments_dir = _resolve_path(paths_cfg.get("segments_dir"))
    if output_dir is None:
        output_dir = _resolve_path(paths_cfg.get("output_base_path"))

    os.makedirs(output_dir, exist_ok=True)
    video_files = sorted(glob.glob(os.path.join(videos_dir, "*.mp4")))
    if not video_files:
        print(f"No videos found in {videos_dir}")
        return [], [], [], []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = setup_model(config.get("model", {}), device)

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
    process_any4d_segments_for_trajectories()


if __name__ == "__main__":
    main()

