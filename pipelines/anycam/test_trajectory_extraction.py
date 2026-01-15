#!/usr/bin/env python3
"""
Test AnyCam trajectory extraction on all segments in a video folder.
"""

import argparse

from extrinsic_path.extract_extrinsic_path import process_anycam_segments_for_trajectories


def main():
    parser = argparse.ArgumentParser(description="AnyCam batch trajectory extraction test")
    parser.add_argument("--config_path", default=None, help="Path to anycam_trajectory_config.yml")
    args = parser.parse_args()

    process_anycam_segments_for_trajectories(args.config_path)


if __name__ == "__main__":
    main()

