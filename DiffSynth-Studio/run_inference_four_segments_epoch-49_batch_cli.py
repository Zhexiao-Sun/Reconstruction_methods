#!/usr/bin/env python3
"""
python run_inference_four_segments_epoch-49_batch_cli.py --no-cpu-offload --cfg-merge --no-vae-tiling
"""
import argparse
import csv
import os
import re
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline


DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"
    "三条腿，背景人很多，倒着走"
)

# ---------------------------------------------------------------------------
# User-configurable settings
# ---------------------------------------------------------------------------
# BASE_DIR = Path("dataset/data_inference")
# METADATA_PATH = Path("dataset/data_inference/metadata.csv")
BASE_DIR = Path("dataset/data_test")
METADATA_PATH = Path("dataset/data_test/metadata.csv")
OUTPUT_DIR = None  # Set to Path("custom/output/dir") to override the automatic folder
LORA_PATH = Path("models/train/Wan2.2-TI2V-5B_lora_four_segments_121_frames/four_segments_epoch-49.safetensors")  # Set to Path("path/to/lora.safetensors") when using LoRA
LORA_ALPHA = 1.0
HEIGHT = 704
WIDTH = 1280
NUM_FRAMES = 121
FPS = 15
QUALITY = 5
SEED = 1
NEGATIVE_PROMPT = DEFAULT_NEGATIVE_PROMPT
WAN_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
TOKENIZER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
LOCAL_MODEL_PATH = Path(os.getenv("WAN_MODELS_DIR", "/openbayes/input/input0/DiffSynth-Studio/models"))


def parse_args():
    parser = argparse.ArgumentParser(description="Wan2.2 TI2V LoRA inference with runtime toggles.")
    parser.add_argument(
        "--cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CPU offload and VRAM management (default: enabled).",
    )
    parser.add_argument(
        "--cfg-merge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable merged CFG (default: disabled).",
    )
    parser.add_argument(
        "--vae-tiling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable VAE tiling (default: enabled).",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of denoising steps (default: 50).",
    )
    return parser.parse_args()


def sanitize(name: str) -> str:
    """Return a filesystem-safe tag derived from the LoRA filename."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def infer_output_dir(explicit: Path | None, lora_path: Path | None, num_frames: int) -> Path:
    """Decide the output folder based on whether LoRA is used."""
    if explicit:
        return explicit
    base = Path("outputs")
    if lora_path:
        tag = sanitize(lora_path.stem)
        return base / f"wan2.2-ti2v-5b_inference_lora_{tag}_{num_frames}_frames"
    return base / f"wan2.2-ti2v-5b_inference_base_{num_frames}_frames"


def build_pipeline(lora_path: Path | None, lora_alpha: float, cpu_offload: bool):
    os.environ.setdefault("MODELSCOPE_OFFLINE", "1")
    offload_device = "cpu" if cpu_offload else None
    tokenizer_config = ModelConfig(
        model_id=TOKENIZER_MODEL_ID,
        origin_file_pattern="google/*",
        local_model_path=str(LOCAL_MODEL_PATH),
        skip_download=True,
    )
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(
                model_id=WAN_MODEL_ID,
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                offload_device=offload_device,
                local_model_path=str(LOCAL_MODEL_PATH),
                skip_download=True,
            ),
            ModelConfig(
                model_id=WAN_MODEL_ID,
                origin_file_pattern="diffusion_pytorch_model*.safetensors",
                offload_device=offload_device,
                local_model_path=str(LOCAL_MODEL_PATH),
                skip_download=True,
            ),
            ModelConfig(
                model_id=WAN_MODEL_ID,
                origin_file_pattern="Wan2.2_VAE.pth",
                offload_device=offload_device,
                local_model_path=str(LOCAL_MODEL_PATH),
                skip_download=True,
            ),
        ],
        tokenizer_config=tokenizer_config,
        redirect_common_files=True,
    )
    if lora_path:
        pipe.load_lora(pipe.dit, str(lora_path), alpha=lora_alpha)
        print(f"[info] LoRA loaded: {lora_path} (alpha={lora_alpha})")
    else:
        print("[info] No LoRA path provided; using base model only.")
    if cpu_offload:
        pipe.enable_vram_management()
    return pipe


def main():
    args = parse_args()
    total_start = time.perf_counter()
    output_dir = infer_output_dir(OUTPUT_DIR, LORA_PATH, NUM_FRAMES)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] Saving outputs to: {output_dir}")
    print(f"[info] Loading Wan models from: {LOCAL_MODEL_PATH}")

    load_start = time.perf_counter()
    pipe = build_pipeline(LORA_PATH, LORA_ALPHA, cpu_offload=args.cpu_offload)
    print(f"[timing] Pipeline load: {time.perf_counter() - load_start:.2f}s")

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        entries = list(csv.DictReader(f))

    for idx, row in enumerate(tqdm(entries, desc="Generating")):
        image_path = BASE_DIR / row["image"]
        if not image_path.exists():
            print(f"[warn] Missing {image_path}, skipping.")
            continue

        input_image = Image.open(image_path).resize((WIDTH, HEIGHT))
        seed = SEED + idx
        output_path = output_dir / f"{Path(row['image']).stem}.mp4"

        infer_start = time.perf_counter()
        video = pipe(
            prompt=row["prompt"],
            negative_prompt=NEGATIVE_PROMPT,
            input_image=input_image,
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            seed=seed,
            tiled=args.vae_tiling,
            cfg_merge=args.cfg_merge,
            num_inference_steps=args.num_inference_steps,  # 运行时可调（例如 30）以测试速度/质量
        )
        infer_elapsed = time.perf_counter() - infer_start

        save_start = time.perf_counter()
        save_video(video, str(output_path), fps=FPS, quality=QUALITY)
        save_elapsed = time.perf_counter() - save_start
        print(
            f"[timing] {output_path.name}: inference={infer_elapsed:.2f}s, "
            f"save={save_elapsed:.2f}s, total={infer_elapsed + save_elapsed:.2f}s"
        )

    print(f"[timing] Script total: {time.perf_counter() - total_start:.2f}s")


if __name__ == "__main__":
    main()

