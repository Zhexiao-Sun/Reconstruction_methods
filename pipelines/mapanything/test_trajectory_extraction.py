#!/usr/bin/env python3
"""
Test script for MapAnything trajectory extraction on all videos in a directory.
"""

import argparse
import os
import sys

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "../.."))
sys.path.append(FILE_DIR)

from extrinsic_path.extract_extrinsic_path import (
    load_config,
    process_mapanything_segments_for_trajectories,
)


def main():
    parser = argparse.ArgumentParser(description="MapAnything batch trajectory extraction test")
    parser.add_argument("--videos_dir", type=str, default=None)
    parser.add_argument("--segments_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    videos_dir = args.videos_dir or os.path.join(PROJECT_ROOT, config["paths"]["videos_dir"])
    segments_dir = args.segments_dir or os.path.join(PROJECT_ROOT, config["paths"]["segments_dir"])
    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, config["paths"]["output_base_path"])

    process_mapanything_segments_for_trajectories(
        videos_dir=videos_dir,
        segments_dir=segments_dir,
        output_dir=output_dir,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()

