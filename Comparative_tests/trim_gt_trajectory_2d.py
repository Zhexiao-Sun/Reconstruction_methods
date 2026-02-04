#!/usr/bin/env python3
"""
Trim gt_trajectory_2d arrays to 9 points for all samples.

This script:
- creates backups for each JSON file
- replaces gt_trajectory_2d with 9 sampled points
- updates both by_model and by_segment sections (if present)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from typing import Dict, List, Any

import numpy as np


DEFAULT_JSONS = [
    "Reconstruction_methods/Comparative_tests/any4d/evaluation_results_any4d_20260117_184404.json",
    "Reconstruction_methods/Comparative_tests/da3/evaluation_results_20260129_185138.json",
    "Reconstruction_methods/Comparative_tests/mapanything/evaluation_results_20260130_181931.json",
    "Reconstruction_methods/Comparative_tests/vggt/evaluation_results_20251113_152448.json",
    "Reconstruction_methods/Comparative_tests/vipe/evaluation_results_20251113_145409.json",
]


def _sample_indices(total: int, num_points: int, start_idx: int = 0) -> List[int]:
    if total <= num_points or start_idx >= total - 1:
        return list(range(start_idx, total))
    return np.linspace(start_idx, total - 1, num_points, dtype=int).tolist()


def _find_origin_index(gt_trajectory_2d: List[List[float]], eps: float = 1e-6) -> int:
    for idx, (x, y) in enumerate(gt_trajectory_2d):
        if abs(x) <= eps and abs(y) <= eps:
            return idx
    return 0


def _trim_gt(gt_trajectory_2d: List[List[float]], num_points: int) -> List[List[float]]:
    if not gt_trajectory_2d:
        return gt_trajectory_2d
    origin_idx = _find_origin_index(gt_trajectory_2d)
    indices = _sample_indices(len(gt_trajectory_2d), num_points, start_idx=origin_idx)
    return [gt_trajectory_2d[i] for i in indices]


def _process_results(data: Dict[str, Any], num_points: int) -> int:
    updated = 0

    by_model = data.get("by_model", {})
    for model_name, segments in by_model.items():
        for segment_name, entry in segments.items():
            if "gt_trajectory_2d" in entry:
                entry["gt_trajectory_2d"] = _trim_gt(entry["gt_trajectory_2d"], num_points)
                updated += 1

    by_segment = data.get("by_segment", {})
    for segment_name, models in by_segment.items():
        for model_name, entry in models.items():
            if "gt_trajectory_2d" in entry:
                entry["gt_trajectory_2d"] = _trim_gt(entry["gt_trajectory_2d"], num_points)
                updated += 1

    return updated


def _infer_gt_video_rename_label(json_path: str) -> str | None:
    """
    Infer the desired display name for gt_video based on file path.

    - vggt -> "VGGT"
    - vipe -> "ViPE"
    """
    norm = json_path.replace("\\", "/").lower()
    if "/vggt/" in norm:
        return "VGGT"
    if "/vipe/" in norm:
        return "ViPE"
    return None


def _keep_only_gt_video_and_rename(data: Dict[str, Any], new_model_name: str) -> Dict[str, int]:
    """
    Remove all non-gt_video models from both `by_model` and `by_segment`,
    and rename the remaining model key + each entry's `model_name` to `new_model_name`.

    Returns counts for reporting.
    """
    counts = {
        "by_model_models_removed": 0,
        "by_segment_models_removed": 0,
        "entries_renamed": 0,
        "segments_kept": 0,
    }

    # by_model: {model_name -> {segment_name -> entry}}
    by_model = data.get("by_model")
    if isinstance(by_model, dict):
        if "gt_video" in by_model and isinstance(by_model["gt_video"], dict):
            gt_segments = by_model["gt_video"]
            counts["by_model_models_removed"] = max(0, len(by_model) - 1)
            data["by_model"] = {new_model_name: gt_segments}
            for _segment_name, entry in gt_segments.items():
                if isinstance(entry, dict):
                    if entry.get("model_name") != new_model_name:
                        entry["model_name"] = new_model_name
                        counts["entries_renamed"] += 1
        else:
            # No gt_video present: clear by_model for consistency with "keep only gt_video".
            counts["by_model_models_removed"] = len(by_model)
            data["by_model"] = {}

    # by_segment: {segment_name -> {model_name -> entry}}
    by_segment = data.get("by_segment")
    if isinstance(by_segment, dict):
        new_by_segment: Dict[str, Any] = {}
        for segment_name, models in by_segment.items():
            if not isinstance(models, dict):
                continue

            if "gt_video" not in models:
                # If segment lacks gt_video, drop it (since we only keep gt_video).
                counts["by_segment_models_removed"] += len(models)
                continue

            gt_entry = models.get("gt_video")
            counts["by_segment_models_removed"] += max(0, len(models) - 1)
            if isinstance(gt_entry, dict):
                if gt_entry.get("model_name") != new_model_name:
                    gt_entry["model_name"] = new_model_name
                    counts["entries_renamed"] += 1
            new_by_segment[segment_name] = {new_model_name: gt_entry}
            counts["segments_kept"] += 1

        data["by_segment"] = new_by_segment

    return counts


def _backup_file(path: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{path}.bak_{timestamp}"
    shutil.copy2(path, backup_path)
    return backup_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Trim gt_trajectory_2d to N points.")
    parser.add_argument(
        "--jsons",
        nargs="*",
        default=DEFAULT_JSONS,
        help="List of JSON files to process (absolute or relative to repo root).",
    )
    parser.add_argument("--num_points", type=int, default=9, help="Number of GT points to keep.")
    parser.add_argument(
        "--clean_vggt_vipe_gt_video_only",
        action="store_true",
        help='For vggt/vipe result JSONs: keep only gt_video, rename to "VGGT"/"ViPE".',
    )
    parser.add_argument(
        "--repo_root",
        default="/home/zhexiao/MA/2.Versuch",
        help="Repository root for resolving relative paths.",
    )
    args = parser.parse_args()

    for rel_path in args.jsons:
        path = rel_path
        if not os.path.isabs(path):
            path = os.path.join(args.repo_root, rel_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"JSON not found: {path}")

        backup_path = _backup_file(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        clean_counts: Dict[str, int] | None = None
        if args.clean_vggt_vipe_gt_video_only:
            new_name = _infer_gt_video_rename_label(path)
            if new_name is not None:
                clean_counts = _keep_only_gt_video_and_rename(data, new_name)

        updated = _process_results(data, args.num_points)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✓ {os.path.basename(path)}: updated {updated} entries")
        if clean_counts is not None:
            print(
                "  cleaned(gt_video only): "
                f"removed_by_model={clean_counts['by_model_models_removed']}, "
                f"removed_by_segment={clean_counts['by_segment_models_removed']}, "
                f"segments_kept={clean_counts['segments_kept']}, "
                f"entries_renamed={clean_counts['entries_renamed']}"
            )
        print(f"  backup: {backup_path}")


if __name__ == "__main__":
    main()
