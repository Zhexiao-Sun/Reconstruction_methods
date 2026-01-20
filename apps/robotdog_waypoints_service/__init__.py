"""
RobotDog 端到端 waypoints 服务包。

该目录提供一个单体同步服务：输入一张当前照片+prompt，先用 Wan2.2-TI2V-5B(LoRA) 生成视频，再用 MapAnything 从视频均匀抽 9 帧并输出 9 个 trajectory_2d waypoints。
"""


