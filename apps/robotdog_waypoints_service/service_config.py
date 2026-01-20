"""
服务配置（通过环境变量控制默认行为）。

该文件定义 RobotDog waypoints 服务的运行参数：路径、Wan 推理参数、MapAnything 抽帧数量等。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    return int(v)


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return default
    return float(v)


@dataclass(frozen=True)
class ServiceConfig:
    # 代码/数据根目录
    reconstruction_root: Path = Path(os.getenv("RECONSTRUCTION_ROOT", "/openbayes/home/Reconstruction_methods"))
    runtime_root: Path = Path(os.getenv("RUNTIME_ROOT", "/openbayes/home/Reconstruction_methods/runtime/requests"))

    # 强制离线（避免访问 huggingface.co / modelscope.cn 等外网）  # 关键：解决服务器无法出网导致的超时
    offline_mode: bool = _env_bool("OFFLINE_MODE", True)

    # torch.hub 缓存目录（DINOv2 等 backbone 会走这里）  # 关键：避免 torch.hub 触网下载/重复下载
    # 注意：torch.hub 的目录通常形如 ".../.torch/hub"（你机器现有缓存也在该位置）。  # 避免误配导致重新下载
    torch_home: Path = Path(os.getenv("TORCH_HOME", "/openbayes/home/.torch/hub"))

    # Wan 推理参数（默认与 run_inference_four_segments_epoch-49_batch_cli.py 保持一致）
    wan_height: int = _env_int("WAN_HEIGHT", 704)
    wan_width: int = _env_int("WAN_WIDTH", 1280)
    wan_num_frames: int = _env_int("WAN_NUM_FRAMES", 121)
    wan_fps: int = _env_int("WAN_FPS", 15)
    wan_quality: int = _env_int("WAN_QUALITY", 5)
    wan_seed: int = _env_int("WAN_SEED", 1)
    wan_num_inference_steps: int = _env_int("WAN_NUM_INFERENCE_STEPS", 50)
    wan_cpu_offload: bool = _env_bool("WAN_CPU_OFFLOAD", False)  # 与你常用启动参数 --no-cpu-offload 对齐
    wan_cfg_merge: bool = _env_bool("WAN_CFG_MERGE", True)  # 与你常用启动参数 --cfg-merge 对齐
    wan_vae_tiling: bool = _env_bool("WAN_VAE_TILING", False)  # 与你常用启动参数 --no-vae-tiling 对齐

    wan_model_id: str = os.getenv("WAN_MODEL_ID", "Wan-AI/Wan2.2-TI2V-5B")
    wan_lora_path: Path = Path(
        os.getenv(
            "WAN_LORA_PATH",
            "/openbayes/home/Reconstruction_methods/DiffSynth-Studio/models/train/"
            "Wan2.2-TI2V-5B_lora_four_segments_121_frames/four_segments_epoch-49.safetensors",
        )
    )
    wan_lora_alpha: float = _env_float("WAN_LORA_ALPHA", 1.0)
    wan_negative_prompt: str = os.getenv(
        "WAN_NEGATIVE_PROMPT",
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
        "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
        "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"
        "三条腿，背景人很多，倒着走",
    )

    # MapAnything 参数
    map_model_id: str = os.getenv("MAP_MODEL_ID", "facebook/map-anything")
    map_num_frames: int = _env_int("MAP_NUM_FRAMES", 9)
    map_frame_stride: int | None = None  # 若设为 None，则使用 num_frames 均匀抽帧

    # MapAnything 本地缓存位置（优先使用本地 snapshot，不走 repo_id 联网检查）  # 关键：避免 huggingface_hub HEAD 请求
    hf_hub_dir: Path = Path(os.getenv("HF_HUB_DIR", "/openbayes/home/huggingface/hub"))
    map_local_snapshot: Path | None = (
        Path(os.getenv("MAP_LOCAL_SNAPSHOT")) if os.getenv("MAP_LOCAL_SNAPSHOT") else None
    )


