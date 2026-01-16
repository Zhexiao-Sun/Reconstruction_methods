#!/usr/bin/env python3
"""
Test script for DynamicVerse trajectory extraction on a single video.
"""

import argparse
import os
import sys

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "../.."))
sys.path.append(FILE_DIR)

from extrinsic_path.extract_extrinsic_path import load_config, process_single_video


def pick_first_video(videos_dir):
    for name in sorted(os.listdir(videos_dir)):
        if name.lower().endswith(".mp4"):
            return os.path.join(videos_dir, name)
    return None


def main():
    parser = argparse.ArgumentParser(description="DynamicVerse single video test")
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--videos_dir", type=str, default=None)
    parser.add_argument("--segments_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    videos_dir = args.videos_dir or os.path.join(PROJECT_ROOT, config["paths"]["videos_dir"])
    segments_dir = args.segments_dir or os.path.join(PROJECT_ROOT, config["paths"]["segments_dir"])
    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, config["paths"]["output_base_path"])

    video_path = args.video_path or pick_first_video(videos_dir)
    if not video_path or not os.path.exists(video_path):
        print(f"No video found in {videos_dir}")
        return

    process_single_video(
        video_path,
        config,
        output_dir,
        segments_dir=segments_dir,
    )


if __name__ == "__main__":
    main()

