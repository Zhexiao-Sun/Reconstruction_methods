"""
API 请求/响应的数据结构定义。

该文件包含 FastAPI 使用的 Pydantic 模型：输入（照片+prompt）与输出（9个 trajectory_2d waypoints）。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class InferWaypointsRequest(BaseModel):
    prompt: str = Field(..., description="导航指令/语义目标描述")
    image_base64: str = Field(..., description="输入图片 base64（建议为 PNG/JPG 的原始字节编码）")

    # 可选覆盖：允许服务端在需要时调整推理速度/质量；默认值在 ServiceConfig 中定义
    num_inference_steps: int | None = Field(None, ge=1, le=200, description="Wan 去噪步数（如 30/50）")
    seed: int | None = Field(None, description="随机种子；若不传则使用默认种子并按请求序号偏移")


class InferWaypointsResponse(BaseModel):
    request_id: str
    trajectory_2d: list[list[float]] = Field(..., description="9个二维轨迹点：[[x,y], ...]")
    video_path: str = Field(..., description="服务端保存的视频路径（用于审计/复现）")
    meta: dict = Field(default_factory=dict, description="耗时/参数等元信息")


