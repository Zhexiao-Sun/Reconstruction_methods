## RobotDog Waypoints Service（Wan → MapAnything → 9点轨迹）

### 功能
- **输入**：一张当前照片（base64）+ `prompt`
- **输出**：`trajectory_2d`（9 个二维轨迹点），并在服务端保存 Wan 生成的 mp4

### 目录结构（关键）
- `app.py`：FastAPI 服务入口
- `wan_runner.py`：Wan2.2-TI2V-5B（LoRA）推理封装
- `mapanything_runner.py`：MapAnything（视频→9帧→9点）推理封装
- `scripts/offline_e2e.py`：离线端到端验证（直接用 mp4 得到 9 点）
- `scripts/offline_full.py`：离线端到端验证（图片+prompt → mp4 → 9 点）

### 环境搭建
请按 `Reconstruction_methods/docs/env_wan_mapanything.md` 搭建统一环境。  # 统一依赖入口

### 离线端到端验证（推荐先跑）
示例：直接对已有视频跑 MapAnything 输出 9 点（最省资源）

```bash
python /openbayes/home/Reconstruction_methods/apps/robotdog_waypoints_service/scripts/offline_e2e.py \
  --video_path /openbayes/home/Reconstruction_methods/DiffSynth-Studio/outputs/wan2.2-ti2v-5b_inference_lora_four_segments_epoch-49_121_frames/sample_083_custom_segment_075_377_current_frame.mp4
```

示例：按真实输入（图片+prompt）跑完整链路（更接近机器人实际调用）

```bash
python /openbayes/home/Reconstruction_methods/apps/robotdog_waypoints_service/scripts/offline_full.py \
  --image_path /openbayes/home/Reconstruction_methods/DiffSynth-Studio/dataset/data_test/sample_083_custom_segment_075_377_current_frame.png \
  --prompt "Move to the plant on the outside corner of the wall, stop in front of it, and avoid any collisions"
```

### 启动服务（单 worker，GPU 串行）

```bash
python -m uvicorn apps.robotdog_waypoints_service.app:app --app-dir /openbayes/home/Reconstruction_methods --host 0.0.0.0 --port 8000 --workers 1
```

### 调用接口
- `POST /infer_waypoints`
  - body：`{"prompt": "...", "image_base64": "..."}`

### hyper.ai Serving 对接建议
- **并发**：建议 1（同步串行，避免显存争用）  # 与当前服务锁一致
- **缓存挂载**：建议挂载 `HF_HOME`/`TRANSFORMERS_CACHE` 到大磁盘目录  # 避免重复拉权重
- **健康检查**：`GET /healthz`


