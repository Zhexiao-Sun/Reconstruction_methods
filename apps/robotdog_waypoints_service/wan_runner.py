"""
Wan2.2-TI2V-5B（LoRA）推理封装。

该文件把 `DiffSynth-Studio/run_inference_four_segments_epoch-49_batch_cli.py` 的核心逻辑封装为可复用类，供离线验证与 API 服务调用：输入一张图+prompt，输出 mp4 视频到指定路径。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image


def _add_diffsynth_to_syspath(reconstruction_root: Path) -> None:
    diffsynth_root = reconstruction_root / "DiffSynth-Studio"
    if str(diffsynth_root) not in sys.path:
        sys.path.insert(0, str(diffsynth_root))  # 让 diffsynth 可被 import（不依赖 pip install -e .）


class WanRunner:
    def __init__(self, cfg: Any):
        _add_diffsynth_to_syspath(cfg.reconstruction_root)

        from diffsynth import save_video  # 延迟 import，避免模块导入时触发不必要的初始化
        from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

        self._save_video = save_video
        self._cfg = cfg

        if getattr(cfg, "offline_mode", False):
            os.environ.setdefault("MODELSCOPE_OFFLINE", "1")  # 禁止 modelscope 联网  # 关键：保证纯本地加载

        local_model_path = str(cfg.reconstruction_root / "DiffSynth-Studio" / "models")  # 使用绝对路径，避免 cwd 影响  # 关键稳定性

        # 关键修复：WanVideoPipeline.from_pretrained 默认会下载 tokenizer_config（Wan2.1-T2V-1.3B 的 google/*），
        # 会导致在离线/网络差环境下额外耗时甚至失败。这里强制只用本地已有 tokenizer 资源。  # 关键：避免重复下载
        tokenizer_config = ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="google/*",
            local_model_path=local_model_path,
            skip_download=True,  # 关键：只查本地，不触发 snapshot_download
        )

        offload_device = "cpu" if cfg.wan_cpu_offload else None  # 与 CLI 行为保持一致
        self._pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(
                    model_id=cfg.wan_model_id,
                    origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                    offload_device=offload_device,
                    local_model_path=local_model_path,
                    skip_download=True,  # 关键：只在本地检查/读取，不触发 snapshot_download
                ),
                ModelConfig(
                    model_id=cfg.wan_model_id,
                    origin_file_pattern="diffusion_pytorch_model*.safetensors",
                    offload_device=offload_device,
                    local_model_path=local_model_path,
                    skip_download=True,
                ),
                ModelConfig(
                    model_id=cfg.wan_model_id,
                    origin_file_pattern="Wan2.2_VAE.pth",
                    offload_device=offload_device,
                    local_model_path=local_model_path,
                    skip_download=True,
                ),
            ],
            tokenizer_config=tokenizer_config,
        )

        if cfg.wan_lora_path and Path(cfg.wan_lora_path).exists():
            self._pipe.load_lora(self._pipe.dit, str(cfg.wan_lora_path), alpha=float(cfg.wan_lora_alpha))
        if cfg.wan_cpu_offload:
            self._pipe.enable_vram_management()

    @torch.no_grad()
    def generate_mp4(
        self,
        *,
        input_image: Image.Image,
        prompt: str,
        output_path: Path,
        seed: int,
        num_inference_steps: int,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        resized = input_image.resize((int(self._cfg.wan_width), int(self._cfg.wan_height)))  # 与原脚本一致
        video = self._pipe(
            prompt=prompt,
            negative_prompt=str(self._cfg.wan_negative_prompt),
            input_image=resized,
            height=int(self._cfg.wan_height),
            width=int(self._cfg.wan_width),
            num_frames=int(self._cfg.wan_num_frames),
            seed=int(seed),
            tiled=bool(self._cfg.wan_vae_tiling),
            cfg_merge=bool(self._cfg.wan_cfg_merge),
            num_inference_steps=int(num_inference_steps),
        )
        self._save_video(video, str(output_path), fps=int(self._cfg.wan_fps), quality=int(self._cfg.wan_quality))


