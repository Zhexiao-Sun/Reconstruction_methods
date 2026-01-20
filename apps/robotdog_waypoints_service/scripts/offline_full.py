"""
python /openbayes/home/Reconstruction_methods/apps/robotdog_waypoints_service/scripts/offline_full.py \
  --image_path /openbayes/home/Reconstruction_methods/DiffSynth-Studio/dataset/data_test/sample_083_custom_segment_075_377_current_frame.png \
  --prompt "Move to the plant on the outside corner of the wall, stop in front of it, and avoid any collisions"

离线端到端验证脚本（图片+prompt → Wan 生成 mp4 → MapAnything 输出 9 点轨迹）。

用途：
- 在不上线 API 的情况下，按真实 pipeline 输入（单帧图+prompt）跑通整条链路。
- 默认会把中间产物落盘：input.png、wan_output.mp4、map_frames/、result.json。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image


def _add_reconstruction_root_to_syspath(reconstruction_root: Path) -> None:
    if str(reconstruction_root) not in sys.path:
        sys.path.insert(0, str(reconstruction_root))  # 允许直接 import apps.robotdog_waypoints_service.*


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--reconstruction_root", type=str, default="/openbayes/home/Reconstruction_methods")
    p.add_argument("--image_path", type=str, required=True, help="输入图片路径（PNG/JPG）")
    p.add_argument("--prompt", type=str, required=True, help="导航/语义指令")
    p.add_argument("--output_dir", type=str, default=None, help="输出目录（默认使用 runtime/requests/{timestamp}）")
    p.add_argument("--seed", type=int, default=None, help="随机种子（默认用 ServiceConfig 的 WAN_SEED）")
    p.add_argument("--num_inference_steps", type=int, default=None, help="Wan 去噪步数（默认用 ServiceConfig 的 WAN_NUM_INFERENCE_STEPS）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    reconstruction_root = Path(args.reconstruction_root)
    _add_reconstruction_root_to_syspath(reconstruction_root)

    from apps.robotdog_waypoints_service.service_config import ServiceConfig
    from apps.robotdog_waypoints_service.utils import ensure_dir, make_request_id
    from apps.robotdog_waypoints_service.wan_runner import WanRunner
    from apps.robotdog_waypoints_service.mapanything_runner import MapAnythingRunner

    cfg = ServiceConfig()

    request_id = make_request_id()
    if args.output_dir:
        out_dir = ensure_dir(Path(args.output_dir))
    else:
        out_dir = ensure_dir(Path(cfg.runtime_root) / request_id)

    input_path = out_dir / "input.png"
    video_path = out_dir / "wan_output.mp4"
    frames_dir = out_dir / "map_frames"
    result_path = out_dir / "result.json"

    img = Image.open(args.image_path).convert("RGB")
    img.save(input_path)

    seed = int(args.seed) if args.seed is not None else int(cfg.wan_seed)
    steps = (
        int(args.num_inference_steps)
        if args.num_inference_steps is not None
        else int(cfg.wan_num_inference_steps)
    )

    t0 = time.time()
    wan = WanRunner(cfg)
    t1 = time.time()
    wan.generate_mp4(
        input_image=img,
        prompt=args.prompt,
        output_path=video_path,
        seed=seed,
        num_inference_steps=steps,
    )
    t2 = time.time()
    map_runner = MapAnythingRunner(cfg)
    t3 = time.time()
    traj2d = map_runner.infer_trajectory_2d_from_video(video_path=video_path, frames_dir=frames_dir)
    t4 = time.time()

    payload = {
        "request_id": request_id,
        "input_path": str(input_path),
        "video_path": str(video_path),
        "trajectory_2d": traj2d,
        "meta": {
            "seed": seed,
            "num_inference_steps": steps,
            "wan_num_frames": int(cfg.wan_num_frames),
            "map_num_frames": int(cfg.map_num_frames),
            "timing_sec": {
                "init_wan": float(t1 - t0),
                "wan": float(t2 - t1),
                "init_mapanything": float(t3 - t2),
                "mapanything": float(t4 - t3),
                "total": float(t4 - t0),
            },
        },
    }
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


