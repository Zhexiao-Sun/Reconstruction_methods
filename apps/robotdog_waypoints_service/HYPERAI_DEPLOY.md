## hyper.ai Serving 部署要点（单体/同步/单并发）

### 1) 入口与启动命令
- **入口文件**：`apps/robotdog_waypoints_service/app.py`
- **启动命令**（建议单 worker）：使用 `apps/robotdog_waypoints_service/start.sh`  # 保证并发=1，避免显存争用

### 2) 推荐环境变量
参考 `apps/robotdog_waypoints_service/config.env.example`，重点是：
- **缓存挂载**：`HF_HOME`、`TRANSFORMERS_CACHE`
- **运行目录**：`RUNTIME_ROOT`（用于保存每次请求的 `input.png / wan_output.mp4 / result.json / map_frames/`）
- **并发**：服务端保持 `WORKERS=1`，并且服务内部有互斥锁确保 GPU 串行  # 稳定性优先

### 3) 健康检查
- `GET /healthz` 返回 `{"ok": true}`  # 供平台探活

### 4) 资源建议
- **GPU**：Wan 推理更吃显存；如果显存不足，优先把 `WAN_CPU_OFFLOAD=1` 或 `WAN_VAE_TILING=1` 打开（会变慢但更稳）。  # 典型 OOM 处理策略

### 5) 输出与审计
每个请求会在 `RUNTIME_ROOT/{request_id}/` 下保存：
- `input.png`：输入图片
- `wan_output.mp4`：生成视频
- `map_frames/`：MapAnything 抽帧缓存（你也可以后续加定期清理）
- `result.json`：包含 `trajectory_2d` 与耗时等元信息


