"""
RobotDog Waypoints 单体同步服务（FastAPI）。

该服务在进程启动时加载 Wan 与 MapAnything 模型；每次请求串行执行：图片+prompt → Wan 生成 mp4 → MapAnything 抽9帧 → 输出9个 trajectory_2d waypoints。
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .schemas import InferWaypointsRequest, InferWaypointsResponse
from .service_config import ServiceConfig
from .utils import decode_base64_image, ensure_dir, make_request_id
from .wan_runner import WanRunner
from .mapanything_runner import MapAnythingRunner


cfg = ServiceConfig()
app = FastAPI(title="robotdog-waypoints-service", version="0.1.0")

_lock = Lock()  # GPU 串行：同步服务下避免并发抢显存/互相影响  # 关键稳定性策略
_wan: WanRunner | None = None
_map: MapAnythingRunner | None = None


@app.on_event("startup")
def _startup() -> None:
    global _wan, _map
    ensure_dir(cfg.runtime_root)
    _wan = WanRunner(cfg)
    _map = MapAnythingRunner(cfg)


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


def _validate_request_id(request_id: str) -> None:
    # 约束 request_id 格式，避免 path traversal 等安全问题。  # 关键安全
    if not re.fullmatch(r"[0-9]{8}_[0-9]{6}_[0-9a-f]{8}", request_id):
        raise HTTPException(status_code=400, detail="Invalid request_id")


@app.get("/requests/{request_id}/result.json")
def get_result_json(request_id: str):
    _validate_request_id(request_id)
    req_dir = Path("/openbayes/home/Reconstruction_methods/runtime/requests") / request_id
    result_path = req_dir / "result.json"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="result.json not found")
    return FileResponse(str(result_path), media_type="application/json")


@app.get("/requests/{request_id}/wan_output.mp4")
def download_wan_output_mp4(request_id: str):
    _validate_request_id(request_id)
    req_dir = Path("/openbayes/home/Reconstruction_methods/runtime/requests") / request_id
    video_path = req_dir / "wan_output.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="wan_output.mp4 not found")
    return FileResponse(str(video_path), media_type="video/mp4", filename=f"{request_id}.mp4")


@app.post("/infer_waypoints", response_model=InferWaypointsResponse)
def infer_waypoints(req: InferWaypointsRequest) -> InferWaypointsResponse:
    if _wan is None or _map is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    request_id = make_request_id()
    req_dir = ensure_dir(Path(cfg.runtime_root) / request_id)
    input_path = req_dir / "input.png"
    video_path = req_dir / "wan_output.mp4"
    frames_dir = req_dir / "map_frames"
    result_path = req_dir / "result.json"

    img = decode_base64_image(req.image_base64)
    img.save(input_path)

    seed = int(req.seed) if req.seed is not None else int(cfg.wan_seed)
    steps = int(req.num_inference_steps) if req.num_inference_steps is not None else int(cfg.wan_num_inference_steps)

    t0 = time.time()
    with _lock:
        t1 = time.time()
        _wan.generate_mp4(
            input_image=img,
            prompt=req.prompt,
            output_path=video_path,
            seed=seed,
            num_inference_steps=steps,
        )
        t2 = time.time()
        traj2d = _map.infer_trajectory_2d_from_video(video_path=video_path, frames_dir=frames_dir)
        t3 = time.time()

    payload = {
        "request_id": request_id,
        "trajectory_2d": traj2d,
        "video_path": str(video_path),
        "meta": {
            "wan_num_frames": int(cfg.wan_num_frames),
            "map_num_frames": int(cfg.map_num_frames),
            "seed": seed,
            "num_inference_steps": steps,
            "timing_sec": {
                "queue_wait": float(t1 - t0),
                "wan": float(t2 - t1),
                "mapanything": float(t3 - t2),
                "total": float(t3 - t0),
            },
        },
    }
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    return InferWaypointsResponse(
        request_id=request_id,
        trajectory_2d=traj2d,
        video_path=str(video_path),
        meta=payload["meta"],
    )


