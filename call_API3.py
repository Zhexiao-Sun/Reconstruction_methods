"""
机器人端调用示例（healthz → infer_waypoints → 打印 timing → 下载 result.json 与 mp4）。

用法：
python /openbayes/home/Reconstruction_methods/call_API.py \
  --service_url "https://xxx.serving.hyperai.host" \
  --image_path "/path/to/input.png" \
  --prompt "Move to ..."
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import requests

SERVICE_URL = "https://ea5i5e07fh1w-s8tvic3pz6qy.serving.hyperai.host"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image_path", type=str, required=True, help="输入图片路径（PNG/JPG）")
    p.add_argument("--prompt", type=str, required=True, help="导航/语义指令")
    p.add_argument("--timeout_infer_sec", type=int, default=900, help="推理超时（秒）")
    p.add_argument("--out_dir", type=str, default=".", help="下载文件保存目录")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    service_url = args.service_url.rstrip("/")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "prompt": args.prompt,
        "image_base64": image_b64,
        # 可选：
        # "num_inference_steps": 50,
        # "seed": 1,
    }

    # 1) healthz
    health = requests.get(f"{service_url}/healthz", timeout=30)
    health.raise_for_status()
    print("healthz:", health.json())

    # 2) infer
    resp = requests.post(
        f"{service_url}/infer_waypoints",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=int(args.timeout_infer_sec),
    )
    resp.raise_for_status()
    data = resp.json()

    trajectory_2d = data["trajectory_2d"]
    request_id = data.get("request_id")
    video_path = data.get("video_path")
    timing_sec = (data.get("meta") or {}).get("timing_sec")

    print("trajectory_2d:", trajectory_2d)
    print("request_id:", request_id)
    print("video_path:", video_path)
    print("timing_sec:", timing_sec)

    if not request_id:
        print("warning: response has no request_id; skip downloading files")
        return

    # 3) 下载 result.json（服务端保存的原始结果）
    rj = requests.get(f"{service_url}/requests/{request_id}/result.json", timeout=60)
    rj.raise_for_status()
    result_obj = rj.json()
    out_json = out_dir / f"{request_id}.result.json"
    out_json.write_text(json.dumps(result_obj, ensure_ascii=False, indent=2))
    print("saved_result_json:", str(out_json))

    # 4) 下载 mp4
    mp4_resp = requests.get(f"{service_url}/requests/{request_id}/wan_output.mp4", stream=True, timeout=900)
    mp4_resp.raise_for_status()
    out_mp4 = out_dir / f"{request_id}.mp4"
    with open(out_mp4, "wb") as f:
        for chunk in mp4_resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    print("saved_mp4:", str(out_mp4))


if __name__ == "__main__":
    main()


