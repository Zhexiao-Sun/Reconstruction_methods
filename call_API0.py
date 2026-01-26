import base64
import json
import requests

SERVICE_URL = "https://ea5i5e07fh1w-s8tvic3pz6qy.serving.hyperai.host"

image_path = "/home/zhexiao/MA/2.Versuch/Reconstruction_methods/dataset/Benchmark/sample_083_custom_segment_075_377/sample_083_custom_segment_075_377_current_frame.png"
prompt = "Move to the plant on the outside corner of the wall, stop in front of it, and avoid any collisions"

with open(image_path, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "prompt": prompt,
    "image_base64": image_b64,
    # 可选：
    # "num_inference_steps": 50,
    # "seed": 1,
}

headers = {
    "Content-Type": "application/json",
}

# 可选：先探活
health = requests.get(f"{SERVICE_URL}/healthz", timeout=30)
health.raise_for_status()
print("healthz:", health.json())

resp = requests.post(
    f"{SERVICE_URL}/infer_waypoints",
    headers=headers,
    data=json.dumps(payload),
    timeout=600,
)
resp.raise_for_status()

data = resp.json()
trajectory_2d = data["trajectory_2d"]
print("trajectory_2d:", trajectory_2d)
print("request_id:", data.get("request_id"))
print("video_path:", data.get("video_path"))

