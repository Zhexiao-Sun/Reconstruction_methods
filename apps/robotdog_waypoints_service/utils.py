"""
通用工具函数。

该文件提供：request_id 生成、base64 解码为图片、目录管理等辅助能力，供离线脚本与 API 服务复用。
"""

from __future__ import annotations

import base64
import io
import os
import time
import uuid
from pathlib import Path

from PIL import Image


def make_request_id() -> str:
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    rnd = uuid.uuid4().hex[:8]
    return f"{ts}_{rnd}"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def decode_base64_image(image_base64: str) -> Image.Image:
    # 允许 data URL 前缀（如 data:image/png;base64,...）；否则按纯 base64 处理。  # 兼容多种客户端格式
    if "," in image_base64 and image_base64.strip().lower().startswith("data:"):
        image_base64 = image_base64.split(",", 1)[1]
    raw = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def safe_write_bytes(path: Path, data: bytes) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


