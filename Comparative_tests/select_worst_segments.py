#!/usr/bin/env python3
"""
Select worst-N segments per model for both evaluations.

Inputs (expected in an eval output dir):
  - up_to_scale_per_segment.csv
  - metric_per_segment.csv

Worst is defined as highest ate_rmse.
For each model, we output:
  - worst_up_to_scale: top-N by up_to_scale ate_rmse, with both up_to_scale + metric scores
  - worst_metric:      top-N by metric ate_rmse, with both metric + up_to_scale scores

  python Reconstruction_methods/Comparative_tests/select_worst_segments.py \
  --eval-dir Reconstruction_methods/Comparative_tests/eval_outputs_auc3_10 \
  --topk 10 \
  --output Reconstruction_methods/Comparative_tests/eval_outputs_auc3_10/worst10_by_model.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class Score:
    ate_rmse: float | None
    rta_mean_deg: float | None
    num_heading: int | None


def _read_scores(csv_path: str) -> Dict[Tuple[str, str], Score]:
    """
    Returns mapping (model, segment) -> Score
    """
    out: Dict[Tuple[str, str], Score] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"model", "segment", "ate_rmse", "rta_mean_deg", "num_heading"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns {sorted(missing)} in {csv_path}")
        for row in reader:
            model = row["model"]
            seg = row["segment"]
            ate = row["ate_rmse"].strip()
            rta = row["rta_mean_deg"].strip()
            num_h = row["num_heading"].strip()
            out[(model, seg)] = Score(
                ate_rmse=float(ate) if ate and ate.lower() != "none" else None,
                rta_mean_deg=float(rta) if rta and rta.lower() != "none" else None,
                num_heading=int(float(num_h)) if num_h and num_h.lower() != "none" else None,
            )
    return out


def _group_by_model(scores: Dict[Tuple[str, str], Score]) -> Dict[str, List[str]]:
    models: Dict[str, List[str]] = {}
    for (model, seg) in scores.keys():
        models.setdefault(model, []).append(seg)
    # stable ordering
    for m in models:
        models[m] = sorted(set(models[m]))
    return dict(sorted(models.items(), key=lambda kv: kv[0].lower()))


def _top_n_worst(
    model: str,
    segments: List[str],
    primary: Dict[Tuple[str, str], Score],
    n: int,
) -> List[str]:
    items = []
    for seg in segments:
        s = primary.get((model, seg))
        if s is None or s.ate_rmse is None:
            continue
        items.append((s.ate_rmse, seg))
    items.sort(key=lambda x: x[0], reverse=True)  # worst = largest error
    return [seg for _, seg in items[:n]]


def _pack_entry(seg: str, up: Score | None, metric: Score | None) -> dict:
    return {
        "segment": seg,
        "up_to_scale": None
        if up is None
        else {
            "ate_rmse": up.ate_rmse,
            "rta_mean_deg": up.rta_mean_deg,
            "num_heading": up.num_heading,
        },
        "metric": None
        if metric is None
        else {
            "ate_rmse": metric.ate_rmse,
            "rta_mean_deg": metric.rta_mean_deg,
            "num_heading": metric.num_heading,
        },
    }


def _resolve_existing_dir(path: str) -> str:
    """
    Resolve an eval-dir path robustly across different CWDs.

    Common user mistake:
    - CWD is already `.../Reconstruction_methods/Comparative_tests/`
    - but they still pass `Reconstruction_methods/Comparative_tests/eval_outputs_xxx`
      which would otherwise become a duplicated absolute path.
    """
    candidates: List[str] = []
    if os.path.isabs(path):
        candidates.append(path)
    else:
        # Relative to current working directory
        candidates.append(os.path.abspath(path))
        # Relative to this script's directory
        candidates.append(os.path.abspath(os.path.join(SCRIPT_DIR, path)))
        # Strip leading folders by taking basename (e.g., eval_outputs_auc3_10)
        candidates.append(os.path.abspath(os.path.join(SCRIPT_DIR, os.path.basename(path))))

    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[0] if candidates else path


def _resolve_output_path(output: str | None, eval_dir: str) -> str:
    if not output:
        return os.path.join(eval_dir, "worst_segments_topk.json")
    cand1 = output if os.path.isabs(output) else os.path.abspath(output)
    parent = os.path.dirname(cand1)
    if parent == "" or os.path.isdir(parent):
        return cand1
    # Fallback: write under eval_dir with requested filename
    return os.path.join(eval_dir, os.path.basename(output))


def main() -> None:
    parser = argparse.ArgumentParser(description="Select worst segments per model.")
    parser.add_argument(
        "--eval-dir",
        required=True,
        help="Evaluation output directory containing per-segment CSVs.",
    )
    parser.add_argument("--topk", type=int, default=10, help="Top-K worst segments per model.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: <eval-dir>/worst_segments_topk.json).",
    )
    args = parser.parse_args()

    eval_dir = _resolve_existing_dir(args.eval_dir)
    up_csv = os.path.join(eval_dir, "up_to_scale_per_segment.csv")
    metric_csv = os.path.join(eval_dir, "metric_per_segment.csv")
    if not os.path.exists(up_csv):
        raise FileNotFoundError(f"Missing: {up_csv}")
    if not os.path.exists(metric_csv):
        raise FileNotFoundError(f"Missing: {metric_csv}")

    up_scores = _read_scores(up_csv)
    metric_scores = _read_scores(metric_csv)

    # Union of models across both CSVs
    models = sorted(
        set([m for (m, _) in up_scores.keys()]) | set([m for (m, _) in metric_scores.keys()]),
        key=lambda s: s.lower(),
    )

    # Collect segments per model (union)
    segs_by_model = {m: set() for m in models}
    for (m, seg) in up_scores.keys():
        segs_by_model.setdefault(m, set()).add(seg)
    for (m, seg) in metric_scores.keys():
        segs_by_model.setdefault(m, set()).add(seg)

    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "eval_dir": eval_dir,
        "topk": int(args.topk),
        "definition": "worst = highest ate_rmse",
        "models": {},
    }

    for m in models:
        segs = sorted(segs_by_model.get(m, set()))
        worst_up = _top_n_worst(m, segs, up_scores, args.topk)
        worst_metric = _top_n_worst(m, segs, metric_scores, args.topk)

        out["models"][m] = {
            "worst_up_to_scale": [
                _pack_entry(seg, up_scores.get((m, seg)), metric_scores.get((m, seg)))
                for seg in worst_up
            ],
            "worst_metric": [
                _pack_entry(seg, up_scores.get((m, seg)), metric_scores.get((m, seg)))
                for seg in worst_metric
            ],
        }

    output_path = _resolve_output_path(args.output, eval_dir)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved: {output_path}")


if __name__ == "__main__":
    main()

