"""
API request/response schemas.

This file defines Pydantic models used by FastAPI: request inputs (image + prompt)
and outputs (9 2D waypoints in `trajectory_2d`).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class InferWaypointsRequest(BaseModel):
    prompt: str = Field(..., description="Navigation instruction / target description")
    image_base64: str = Field(..., description="Input image in base64 (PNG/JPG bytes)")

    # Optional overrides: allow tuning speed/quality per request; defaults live in ServiceConfig.
    num_inference_steps: int | None = Field(None, ge=1, le=200, description="Wan denoising steps (e.g. 30/50)")
    seed: int | None = Field(None, description="Random seed; if omitted, use default and offset by request index")


class InferWaypointsResponse(BaseModel):
    request_id: str
    trajectory_2d: list[list[float]] = Field(..., description="9 2D points: [[x, y], ...]")
    video_path: str = Field(..., description="Saved video path on the server")
    meta: dict = Field(default_factory=dict, description="Metadata (timing, params, etc.)")


