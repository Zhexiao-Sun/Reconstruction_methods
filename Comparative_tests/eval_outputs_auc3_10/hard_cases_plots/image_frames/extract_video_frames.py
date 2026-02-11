#!/usr/bin/env python3
"""
从视频中提取9帧图像：首帧、尾帧和中间均匀分布的7帧
"""

import cv2
import os
import sys
from pathlib import Path


def extract_frames(video_path, output_dir):
    """
    从视频中提取9帧图像
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录路径
    """
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在: {video_path}")
        return False
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件: {video_path}")
        return False
    
    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames == 0:
        print(f"错误：视频文件没有帧: {video_path}")
        cap.release()
        return False
    
    print(f"视频总帧数: {total_frames}, FPS: {fps:.2f}")
    
    # 确定要提取的帧索引
    # 9帧：首帧(0)、尾帧(total_frames-1)、中间均匀分布的7帧
    if total_frames <= 9:
        # 如果总帧数少于9帧，提取所有帧
        frame_indices = list(range(total_frames))
    else:
        # 计算中间7帧的索引（均匀分布）
        # 在首帧和尾帧之间均匀分布
        step = (total_frames - 1) / 8  # 分成8段，得到9个点（包括首尾）
        frame_indices = [0]  # 首帧
        for i in range(1, 8):  # 中间7帧
            frame_idx = int(i * step)
            frame_indices.append(frame_idx)
        frame_indices.append(total_frames - 1)  # 尾帧
    
    # 获取视频文件名（不含扩展名）作为sample名字
    video_name = Path(video_path).stem
    
    # 提取每一帧
    saved_count = 0
    for idx, frame_num in enumerate(frame_indices):
        # 设置视频位置到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            print(f"警告：无法读取第 {frame_num} 帧")
            continue
        
        # 生成输出文件名
        # 格式：sample_XXX_custom_segment_YYY_ZZZ_frame_00.png
        output_filename = f"{video_name}_frame_{idx:02d}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        # 保存图像
        cv2.imwrite(output_path, frame)
        saved_count += 1
        print(f"已保存: {output_filename} (帧 {frame_num}/{total_frames-1})")
    
    cap.release()
    print(f"\n成功提取 {saved_count} 帧图像到: {output_dir}")
    return True


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python extract_video_frames.py <视频文件路径>")
        print("\n示例:")
        print("  python extract_video_frames.py ../dataset/wm_videos/gt_video/sample_083_custom_segment_075_377.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # 输出目录（脚本所在目录）
    script_dir = Path(__file__).parent
    output_dir = script_dir
    
    # 提取帧
    success = extract_frames(video_path, output_dir)
    
    if success:
        print("\n完成！")
    else:
        print("\n失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()
