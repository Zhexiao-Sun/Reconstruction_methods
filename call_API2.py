import base64
import json
import os  # 添加 os 模块
import requests

SERVICE_URL = "https://ea5i5e07fh1w-s8tvic3pz6qy.serving.hyperai.host"

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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

# 拉 result.json（服务端保存的原始结果）
rj = requests.get(f"{SERVICE_URL}/requests/{data['request_id']}/result.json", timeout=30)
rj.raise_for_status()
result_json_data = rj.json()
print("server_result_json:", result_json_data)

# 保存 result.json 到脚本所在目录
result_json_path = os.path.join(SCRIPT_DIR, f"{data['request_id']}.json")
with open(result_json_path, "w", encoding="utf-8") as f:
    json.dump(result_json_data, f, indent=2, ensure_ascii=False)
print("saved_result_json:", result_json_path)

# 下载 mp4
mp4_resp = requests.get(f"{SERVICE_URL}/requests/{data['request_id']}/wan_output.mp4", stream=True, timeout=600)
mp4_resp.raise_for_status()

# 保存 mp4 到脚本所在目录
out_mp4 = os.path.join(SCRIPT_DIR, f"{data['request_id']}.mp4")
with open(out_mp4, "wb") as f:
    for chunk in mp4_resp.iter_content(chunk_size=1024 * 1024):
        if chunk:
            f.write(chunk)
print("saved_mp4:", out_mp4)