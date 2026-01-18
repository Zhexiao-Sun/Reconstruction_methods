"""
Extract camera trajectories from videos using DynamicVerse image-only inference.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../.."))

# 注意：不要在多视频评测中用全局“已完成”标志缓存预处理结果；否则只会对第一个视频生效，后续视频会被错误跳过。  # noqa: E501


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
            "pipelines/dynamicverse/configs/dynamicverse_trajectory_config.yml",
        )
    if not os.path.exists(config_path):
        return {
            "paths": {
                "videos_dir": "dataset/wm_videos/DynamicVerse",
                "segments_dir": "dataset/Benchmark",
                "output_base_path": "pipelines/dynamicverse/output",
                "workdir": "pipelines/dynamicverse/workdir",
                "dynamicverse_root": "models/DynamicVerse",
                "dynamicba_config": "models/DynamicVerse/dynamicBA/config/config_demo.yaml",
            },
            "preprocess": {
                "frame_stride": 1,
                "max_frames": None,
                "num_frames": None,
                "image_ext": "jpg",
                "unidepth": {"use_v2": False, "use_gt_K": False},
                "cotracker": {"interval": 10, "grid_size": 50},
            },
            "dynamicba": {
                "experiment_name": "base",
                "suffix": "_dynamic_flow_fused",
                "opt_intrinsics": True,
                "depth_dir": "unidepth",
                "cotracker_path": "cotrackerv3_10_50",
                "dyn_mask_dir": "deva",
                "mask_name": "Annotations",
                "max_frames": None,
            },
            "axis_mapping": {"order": ["x", "z", "y"]},
            "visualization": {"save_plots": True, "dpi": 300},
            "output": {"save_npy": True, "save_csv": True, "save_json": True},
        }
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _build_env(dynamicverse_root):
    env = os.environ.copy()
    extra_paths = [
        dynamicverse_root,
        os.path.join(dynamicverse_root, "UniDepth"),
    ]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join([p for p in extra_paths + [existing] if p])
    return env


def _derive_unidepth_dir_name(unidepth_cfg):
    if unidepth_cfg.get("use_gt_K"):
        return "unidepth_gt_K"
    return "unidepth"


def _derive_cotracker_dir_name(cotracker_cfg):
    interval = int(cotracker_cfg.get("interval", 10))
    grid_size = int(cotracker_cfg.get("grid_size", 50))
    return f"cotrackerv3_{interval}_{grid_size}"


def extract_video_frames(video_path, output_dir, frame_stride=1, max_frames=None, num_frames=None, image_ext="jpg"):
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
        frame_path = output_dir / f"frame_{saved_idx:06d}.{image_ext}"
        cv2.imwrite(str(frame_path), frame)
        saved_idx += 1
    cap.release()
    return saved_idx


def _ensure_rgb_frames(video_path, rgb_dir, preprocess_cfg):
    rgb_dir = Path(rgb_dir)
    image_ext = preprocess_cfg.get("image_ext", "jpg")
    existing_frames = list(rgb_dir.glob("*.jpg")) + list(rgb_dir.glob("*.png"))
    desired_num_frames = preprocess_cfg.get("num_frames")
    if desired_num_frames is not None:
        needs_refresh = len(existing_frames) != int(desired_num_frames)
    else:
        needs_refresh = len(existing_frames) == 0
    if needs_refresh:
        if rgb_dir.exists():
            shutil.rmtree(rgb_dir)
        extract_video_frames(
            video_path,
            rgb_dir,
            frame_stride=preprocess_cfg.get("frame_stride", 1),
            max_frames=preprocess_cfg.get("max_frames"),
            num_frames=desired_num_frames,
            image_ext=image_ext,
        )


def _list_scene_dirs(workdir):
    if not os.path.exists(workdir):
        return []
    return [d for d in os.listdir(workdir) if os.path.isdir(os.path.join(workdir, d))]


def _ensure_unidepth(workdir, dynamicverse_root, preprocess_cfg):
    unidepth_cfg = preprocess_cfg.get("unidepth", {})
    output_dir_name = _derive_unidepth_dir_name(unidepth_cfg)
    missing_depth = False
    for scene in _list_scene_dirs(workdir):
        depth_path = os.path.join(workdir, scene, output_dir_name, "depth.npy")
        if not os.path.exists(depth_path):
            missing_depth = True
            break
    if not missing_depth:
        return
    script_path = os.path.join(dynamicverse_root, "preprocess", "run_unidepth.py")
    cmd = [sys.executable, script_path, "--workdir", workdir]
    if unidepth_cfg.get("use_v2"):
        cmd.append("--v2")
    if unidepth_cfg.get("use_gt_K"):
        cmd.append("--use_gt_K")
    subprocess.run(cmd, check=True, cwd=dynamicverse_root, env=_build_env(dynamicverse_root))


def _ensure_cotracker(workdir, dynamicverse_root, preprocess_cfg):
    cotracker_cfg = preprocess_cfg.get("cotracker", {})
    output_dir_name = _derive_cotracker_dir_name(cotracker_cfg)
    missing_tracks = False
    for scene in _list_scene_dirs(workdir):
        results_path = os.path.join(workdir, scene, output_dir_name, "results.npz")
        if not os.path.exists(results_path):
            missing_tracks = True
            break
    if not missing_tracks:
        return
    script_path = os.path.join(dynamicverse_root, "preprocess", "run_cotracker.py")
    cmd = [
        sys.executable,
        script_path,
        "--workdir",
        workdir,
        "--interval",
        str(cotracker_cfg.get("interval", 10)),
        "--grid_size",
        str(cotracker_cfg.get("grid_size", 50)),
    ]
    subprocess.run(cmd, check=True, cwd=dynamicverse_root, env=_build_env(dynamicverse_root))


def _find_pose_file(output_dir, suffix):
    if suffix is None:
        suffix = ""
    direct_path = os.path.join(output_dir, f"poses{suffix}.npz")
    if os.path.exists(direct_path):
        return direct_path
    fallback_path = os.path.join(output_dir, "poses.npz")
    if os.path.exists(fallback_path):
        return fallback_path
    candidates = glob.glob(os.path.join(output_dir, "poses*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No pose file found in {output_dir}")
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _run_dynamicba(workdir, dynamicverse_root, video_name, dynamicba_cfg, config_path, depth_dir_name, cotracker_dir_name):
    experiment_name = dynamicba_cfg.get("experiment_name", "base")
    suffix = dynamicba_cfg.get("suffix", "_dynamic_flow_fused")
    output_dir = os.path.join(workdir, video_name, "dynamicBA", experiment_name)
    pose_file = None
    if os.path.exists(output_dir):
        try:
            pose_file = _find_pose_file(output_dir, suffix)
        except FileNotFoundError:
            pose_file = None
    if pose_file is not None:
        return output_dir, pose_file
    script_path = os.path.join(dynamicverse_root, "dynamicBA", "run.py")
    cmd = [
        sys.executable,
        script_path,
        "--config",
        config_path,
        "--workdir",
        workdir,
        "--video",
        video_name,
        "--experiment_name",
        experiment_name,
        "--depth_dir",
        depth_dir_name,
        "--cotracker_path",
        cotracker_dir_name,
        "--dyn_mask_dir",
        dynamicba_cfg.get("dyn_mask_dir", "deva"),
        "--mask_name",
        dynamicba_cfg.get("mask_name", "Annotations"),
        "--suffix",
        suffix,
    ]
    if dynamicba_cfg.get("opt_intrinsics", True):
        cmd.append("--opt_intrinsics")
    if dynamicba_cfg.get("max_frames") is not None:
        cmd.extend(["--max_frames", str(dynamicba_cfg["max_frames"])])
    subprocess.run(cmd, check=True, cwd=dynamicverse_root, env=_build_env(dynamicverse_root))
    pose_file = _find_pose_file(output_dir, suffix)
    return output_dir, pose_file


def _load_camera_poses(pose_file):
    data = np.load(pose_file)
    if "poses" not in data:
        raise ValueError(f"'poses' not found in {pose_file}")
    poses = data["poses"]
    if poses.ndim == 4:
        poses = poses[0]
    return poses


def _apply_axis_mapping(positions, axis_order):
    axis_map = {"x": 0, "y": 1, "z": 2}
    indices = [axis_map[a] for a in axis_order]
    return positions[:, indices]


def extract_trajectory_from_camera_poses(camera_poses, axis_order=None):
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


def save_trajectory_data(trajectory, orientations, output_path, metadata=None, save_npy=True, save_csv=True, save_json=True):
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


def process_single_video(video_path, config, output_dir, segments_dir=None):
    segment_name = Path(video_path).stem
    paths_cfg = config.get("paths", {})
    preprocess_cfg = config.get("preprocess", {})
    dynamicba_cfg = config.get("dynamicba", {})
    axis_order = config.get("axis_mapping", {}).get("order", ["x", "z", "y"])
    dynamicverse_root = _resolve_path(paths_cfg.get("dynamicverse_root"))
    workdir = _resolve_path(paths_cfg.get("workdir"))
    config_path = _resolve_path(paths_cfg.get("dynamicba_config"))
    os.makedirs(output_dir, exist_ok=True)

    if not dynamicverse_root or not os.path.exists(dynamicverse_root):
        raise FileNotFoundError("DynamicVerse root not found; check config paths.dynamicverse_root")
    if not workdir:
        raise ValueError("workdir is required in config paths.workdir")
    if not config_path or not os.path.exists(config_path):
        raise FileNotFoundError("DynamicBA config not found; check config paths.dynamicba_config")

    scene_dir = os.path.join(workdir, segment_name)
    rgb_dir = os.path.join(scene_dir, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)
    _ensure_rgb_frames(video_path, rgb_dir, preprocess_cfg)

    _ensure_unidepth(workdir, dynamicverse_root, preprocess_cfg)
    _ensure_cotracker(workdir, dynamicverse_root, preprocess_cfg)

    unidepth_dir_name = _derive_unidepth_dir_name(preprocess_cfg.get("unidepth", {}))
    cotracker_dir_name = _derive_cotracker_dir_name(preprocess_cfg.get("cotracker", {}))
    dynamicba_output_dir, pose_file = _run_dynamicba(
        workdir,
        dynamicverse_root,
        segment_name,
        dynamicba_cfg,
        config_path,
        unidepth_dir_name,
        cotracker_dir_name,
    )

    camera_poses = _load_camera_poses(pose_file)
    trajectory, orientations, trajectory_2d = extract_trajectory_from_camera_poses(camera_poses, axis_order=axis_order)
    trajectory_length_meters = float(np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)) if len(trajectory) > 1 else 0.0)

    trajectory_output_dir = Path(output_dir) / "trajectories"
    segment_output_path = trajectory_output_dir / segment_name
    save_trajectory_data(
        trajectory,
        orientations,
        str(segment_output_path),
        metadata={
            "segment_name": segment_name,
            "num_frames": len(trajectory),
            "pose_file": pose_file,
            "dynamicba_output_dir": dynamicba_output_dir,
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
            title=f"DynamicVerse - {segment_name}",
            dpi=config.get("visualization", {}).get("dpi", 300),
        )

    info = {
        "segment_name": segment_name,
        "num_frames": len(trajectory),
        "trajectory_length_meters": trajectory_length_meters,
    }
    return trajectory, orientations, trajectory_2d, info


def process_dynamicverse_segments_for_trajectories(videos_dir=None, segments_dir=None, output_dir=None, config_path=None):
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

    all_trajectories = []
    all_orientations = []
    all_trajectories_2d = []
    segment_info = []

    for video_path in video_files:
        print(f"Processing: {os.path.basename(video_path)}")
        traj, orient, traj_2d, info = process_single_video(
            video_path,
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
    process_dynamicverse_segments_for_trajectories()


if __name__ == "__main__":
    main()

