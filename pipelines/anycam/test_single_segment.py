#!/usr/bin/env python3
"""
Test AnyCam trajectory extraction on a single segment.
"""

import argparse

from extrinsic_path.extract_extrinsic_path import process_single_segment_demo


def main():
    parser = argparse.ArgumentParser(description="AnyCam single-segment test")
    parser.add_argument("--segment_name", required=True, help="Segment name (e.g. sample_090_custom_segment_053_354)")
    parser.add_argument("--config_path", default=None, help="Path to anycam_trajectory_config.yml")
    args = parser.parse_args()

    process_single_segment_demo(args.segment_name, args.config_path)


if __name__ == "__main__":
    main()

