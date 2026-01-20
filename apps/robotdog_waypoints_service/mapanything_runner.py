"""
MapAnything（视频→9帧→9点轨迹）推理封装。

该文件复用 `pipelines/mapanything/extrinsic_path/extract_extrinsic_path.py` 的核心函数，但不做评测/画图/与GT对比；只输出 `trajectory_2d` 的 9 个点。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch


def _add_mapanything_to_syspath(reconstruction_root: Path) -> None:
    map_root = reconstruction_root / "models" / "map-anything"
    if str(map_root) not in sys.path:
        sys.path.insert(0, str(map_root))  # 让 mapanything 可被 import（不依赖 pip install -e .）


def _add_map_pipeline_to_syspath(reconstruction_root: Path) -> None:
    pipe_root = reconstruction_root / "pipelines" / "mapanything"
    if str(pipe_root) not in sys.path:
        sys.path.insert(0, str(pipe_root))  # 让 extrinsic_path.* 可被 import


def _resolve_local_mapanything_snapshot(cfg: Any) -> Path:
    # 用户显式指定 snapshot（推荐：部署时固定到某个 hash）  # 便于可复现
    p = getattr(cfg, "map_local_snapshot", None)
    if p:
        p = Path(p)
        if (p / "config.json").exists() and (p / "model.safetensors").exists():
            return p
        raise FileNotFoundError(f"MAP_LOCAL_SNAPSHOT 指向的目录不完整：{p}")

    hub = Path(getattr(cfg, "hf_hub_dir", "/openbayes/home/huggingface/hub"))
    snaps_root = hub / "models--facebook--map-anything" / "snapshots"
    if not snaps_root.exists():
        raise FileNotFoundError(f"找不到本地 MapAnything snapshots：{snaps_root}")

    snaps = sorted(snaps_root.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
    for s in snaps:
        if (s / "config.json").exists() and (s / "model.safetensors").exists():
            return s
    raise FileNotFoundError(f"未找到包含 config.json + model.safetensors 的 snapshot：{snaps_root}")


def _configure_torchhub_dinov2_offline(cfg: Any) -> None:
    # MapAnything/UniCeption 会在 encoder 里调用 torch.hub.load("facebookresearch/dinov2", ...)  # 关键背景：默认可能触发网络
    # 这里固定 TORCH_HOME，并把 dinov2 的 torch.hub.load 重定向到本地 repo，确保离线可运行。  # 关键：不触网
    torch_home = Path(getattr(cfg, "torch_home", "/openbayes/home/.torch/hub"))
    os.environ.setdefault("TORCH_HOME", str(torch_home))  # 允许外部统一配置  # 便于部署挂载
    torch.hub.set_dir(str(torch_home))  # torch.hub 目录应指向 ".../.torch/hub"  # 避免误配导致重新下载

    if not getattr(cfg, "offline_mode", False):
        return

    # 离线模式：要求 dinov2 权重必须已存在，否则直接报错（避免隐式联网下载）  # 关键：失败要早
    ckpt = torch_home / "checkpoints" / "dinov2_vitg14_pretrain.pth"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"离线模式下缺少 DINOv2 权重：{ckpt}；请先在可联网环境填充 torch.hub 缓存后再部署"
        )

    # dinov2 repo 本地缓存路径（torch.hub clone 的 repo）  # 关键：避免从 github 拉代码
    local_repo = torch_home / "facebookresearch_dinov2_main"
    if not local_repo.exists():
        raise FileNotFoundError(
            f"离线模式下缺少 DINOv2 repo 缓存：{local_repo}；请先在可联网环境跑一次 torch.hub.load 填充缓存"
        )

    real_torchhub_load = torch.hub.load

    def _patched_load(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir == "facebookresearch/dinov2":
            # 强制使用本地 repo（source=local），避免 torch hub 触发网络访问  # 关键：离线稳定
            kwargs.setdefault("source", "local")
            return real_torchhub_load(str(local_repo), model, *args, **kwargs)
        return real_torchhub_load(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = _patched_load


class MapAnythingRunner:
    def __init__(self, cfg: Any):
        _add_mapanything_to_syspath(cfg.reconstruction_root)
        _add_map_pipeline_to_syspath(cfg.reconstruction_root)

        from extrinsic_path.extract_extrinsic_path import (
            extract_video_frames,
            run_mapanything_inference,
            collect_camera_poses,
            extract_trajectory_from_camera_poses,
        )
        from mapanything.models import MapAnything

        self._cfg = cfg
        self._extract_video_frames = extract_video_frames
        self._run_mapanything_inference = run_mapanything_inference
        self._collect_camera_poses = collect_camera_poses
        self._extract_trajectory_from_camera_poses = extract_trajectory_from_camera_poses

        if getattr(cfg, "offline_mode", False):
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

        _configure_torchhub_dinov2_offline(cfg)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        local_snapshot = _resolve_local_mapanything_snapshot(cfg)
        self._model = MapAnything.from_pretrained(str(local_snapshot), local_files_only=True).to(device)  # 关键：只读本地，不访问 huggingface.co
        self._model.eval()

    @torch.no_grad()
    def infer_trajectory_2d_from_video(
        self,
        *,
        video_path: Path,
        frames_dir: Path,
    ) -> list[list[float]]:
        frames_dir.mkdir(parents=True, exist_ok=True)
        self._extract_video_frames(
            str(video_path),
            str(frames_dir),
            frame_stride=self._cfg.map_frame_stride,
            max_frames=None,
            num_frames=int(self._cfg.map_num_frames),
        )
        outputs = self._run_mapanything_inference(frames_dir, self._model, {
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
        camera_poses, _metric_scaling_factors = self._collect_camera_poses(outputs)
        _traj3d, _orient, traj2d = self._extract_trajectory_from_camera_poses(camera_poses)
        return traj2d.tolist()


