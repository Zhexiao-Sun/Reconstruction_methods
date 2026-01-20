"""
python /openbayes/home/Reconstruction_methods/apps/robotdog_waypoints_service/scripts/offline_e2e.py   --video_path /openbayes/home/Reconstruction_methods/DiffSynth-Studio/outputs/wan2.2-ti2v-5b_inference_lora_four_segments_epoch-49_121_frames/sample_083_custom_segment_075_377_current_frame.mp4

离线端到端验证脚本（Wan 输出 mp4 → MapAnything 输出 9 点轨迹）。

该脚本用于在不上线 API 的情况下验证流水线可跑通：可选择直接使用已有 mp4（推荐），或从图片+prompt 调用 Wan 生成 mp4 后再执行 MapAnything。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image


def _add_paths(reconstruction_root: Path) -> None:
    diffsynth_root = reconstruction_root / "DiffSynth-Studio"
    map_root = reconstruction_root / "models" / "map-anything"
    map_pipe_root = reconstruction_root / "pipelines" / "mapanything"
    for p in (diffsynth_root, map_root, map_pipe_root):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))  # 直接从源码目录导入，避免依赖 pip install -e .


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--reconstruction_root", type=str, default="/openbayes/home/Reconstruction_methods")
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--image_path", type=str, default=None)
    p.add_argument("--video_path", type=str, required=True, help="输入 mp4（可直接用 Wan 已生成的视频）")
    p.add_argument("--frames_dir", type=str, default=None, help="抽帧缓存目录（默认在 video 同目录下创建）")
    p.add_argument("--map_snapshot", type=str, default=None, help="本地 MapAnything snapshot 目录（包含 config.json + model.safetensors）")
    return p.parse_args()


def _resolve_local_mapanything_snapshot(map_snapshot: str | None) -> Path:
    if map_snapshot:
        p = Path(map_snapshot)
        if (p / "config.json").exists() and (p / "model.safetensors").exists():
            return p
        raise FileNotFoundError(f"--map_snapshot 指向的目录不完整：{p}")

    snaps_root = Path("/openbayes/home/huggingface/hub/models--facebook--map-anything/snapshots")
    if not snaps_root.exists():
        raise FileNotFoundError(f"找不到本地 MapAnything snapshots：{snaps_root}")
    snaps = sorted(snaps_root.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
    for s in snaps:
        if (s / "config.json").exists() and (s / "model.safetensors").exists():
            return s
    raise FileNotFoundError(f"未找到包含 config.json + model.safetensors 的 snapshot：{snaps_root}")


def main() -> None:
    args = parse_args()
    reconstruction_root = Path(args.reconstruction_root)
    _add_paths(reconstruction_root)

    from extrinsic_path.extract_extrinsic_path import (
        extract_video_frames,
        run_mapanything_inference,
        collect_camera_poses,
        extract_trajectory_from_camera_poses,
    )
    from mapanything.models import MapAnything
    import torch

    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 禁止 huggingface_hub 联网  # 关键：避免 connect timeout
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TORCH_HOME", "/openbayes/home/.torch/hub")  # 固定 torch.hub 缓存目录  # 关键：避免误配导致重新下载
    torch.hub.set_dir(os.environ["TORCH_HOME"])

    # 离线：把 torch.hub.load("facebookresearch/dinov2", ...) 强制重定向到本地 repo  # 关键：避免 github 访问
    local_repo = Path(os.environ["TORCH_HOME"]) / "facebookresearch_dinov2_main"
    ckpt = Path(os.environ["TORCH_HOME"]) / "checkpoints" / "dinov2_vitg14_pretrain.pth"
    if not local_repo.exists():
        raise FileNotFoundError(f"缺少本地 DINOv2 repo 缓存：{local_repo}")
    if not ckpt.exists():
        raise FileNotFoundError(f"缺少本地 DINOv2 权重缓存：{ckpt}")

    real_torchhub_load = torch.hub.load

    def _patched_load(repo_or_dir, model, *a, **kw):
        if repo_or_dir == "facebookresearch/dinov2":
            kw.setdefault("source", "local")
            return real_torchhub_load(str(local_repo), model, *a, **kw)
        return real_torchhub_load(repo_or_dir, model, *a, **kw)

    torch.hub.load = _patched_load

    video_path = Path(args.video_path)
    frames_dir = Path(args.frames_dir) if args.frames_dir else (video_path.parent / f"{video_path.stem}_frames9")
    frames_dir.mkdir(parents=True, exist_ok=True)

    extract_video_frames(str(video_path), str(frames_dir), frame_stride=None, max_frames=None, num_frames=9)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_snapshot = _resolve_local_mapanything_snapshot(args.map_snapshot)
    model = MapAnything.from_pretrained(str(local_snapshot), local_files_only=True).to(device)  # 关键：只读本地 snapshot
    model.eval()

    with torch.no_grad():
        outputs = run_mapanything_inference(frames_dir, model, {
            "resize_mode": "fixed_mapping",
            "resolution_set": 518,
            "memory_efficient_inference": False,
            "use_amp": True,
            "amp_dtype": "bf16",
            "apply_mask": True,
            "mask_edges": True,
            "apply_confidence_mask": False,
            "confidence_percentile": 10,
        })

    camera_poses, _metric_scaling_factors = collect_camera_poses(outputs)
    _traj3d, _orient, traj2d = extract_trajectory_from_camera_poses(camera_poses)
    payload = {"video_path": str(video_path), "trajectory_2d": traj2d.tolist()}
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


