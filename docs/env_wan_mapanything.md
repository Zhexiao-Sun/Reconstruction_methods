# Wan2.2-TI2V-5B（DiffSynth-Studio）+ MapAnything（9点轨迹）统一环境搭建

本文档用于在同一个 Python 环境中同时跑通：

- Wan2.2-TI2V-5B（LoRA）图生视频推理（输出 mp4）
- MapAnything 对 mp4 均匀抽 9 帧并输出 9 个 `trajectory_2d` waypoints
- 后续 API 服务（FastAPI）封装

## 0. 前置条件（必须）

- **Python**：建议 `3.10`（`models/map-anything/pyproject.toml` 要求 `>=3.10`）
- **GPU**：建议 NVIDIA GPU；Wan 推理在 `run_inference_four_segments_epoch-49_batch_cli.py` 里使用 `torch_dtype=bfloat16`，需要较新的 GPU/驱动支持 BF16（若硬件不支持，需后续改成 fp16/float32 再评估速度/显存）。  # 关键硬件约束说明
- **系统工具**：建议安装 `ffmpeg`（用于 mp4 读写/编解码链路更稳）。  # 避免视频读写失败

## 1. 建议的缓存目录（避免重复下载权重）

建议把 HuggingFace/Transformers 缓存放到大磁盘分区：

```bash
export HF_HOME=/openbayes/home/.cache/huggingface
export TRANSFORMERS_CACHE=/openbayes/home/.cache/huggingface/transformers
```

## 2. 创建统一 Conda 环境（推荐）

```bash
conda create -n wan_mapanything python=3.10 -y
conda activate wan_mapanything
```

## 3. 安装 PyTorch（按你的 CUDA 版本选择）

> 这里**不要**盲目 `pip install torch`，请按服务器 CUDA/驱动匹配官方 wheel。  # 避免 CUDA/torch 不匹配

示例（仅示意，请替换成你实际 CUDA 版本对应命令）：

```bash
# 例：CUDA 12.4
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## 4. 安装 DiffSynth-Studio（Wan 推理依赖）

```bash
cd /openbayes/home/Reconstruction_methods/DiffSynth-Studio
python -m pip install -r requirements.txt
python -m pip install -e .
```

## 5. 安装 MapAnything（图像/视频轨迹推理依赖）

```bash
cd /openbayes/home/Reconstruction_methods/models/map-anything
python -m pip install -e .
```

## 6. 安装 API 服务依赖（FastAPI）

```bash
python -m pip install -r /openbayes/home/Reconstruction_methods/apps/robotdog_waypoints_service/requirements.txt
```

## 7. 离线 sanity check（只验证依赖/导入，不触发权重下载）

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)

import diffsynth
print("diffsynth:", getattr(diffsynth, "__version__", "unknown"))

from mapanything.models import MapAnything
print("mapanything: import ok")
PY
```