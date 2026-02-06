#!/usr/bin/env python3
"""
Extract final per-model scores from trajectory_metrics_summary.json.

Input:  trajectory_metrics_summary.json produced by evaluate_trajectory_metrics.py
Output: a compact final_scores.json for paper tables.

python /home/zhexiao/MA/2.Versuch/Reconstruction_methods/Comparative_tests/extract_final_scores.py \
  --input /home/zhexiao/MA/2.Versuch/Reconstruction_methods/Comparative_tests/eval_outputs_au3/trajectory_metrics_summary.json \
  --auc-deg 3 10
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class Score:
    ate_mean: Optional[float]
    auc: Dict[str, Optional[float]]
    auc_pct: Dict[str, Optional[float]]


def _resolve_path(p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(SCRIPT_DIR, p))


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_latest_summary_json(root_dir: str) -> str:
    """
    Find latest Comparative_tests/eval_outputs_*/trajectory_metrics_summary.json by dir name.
    """
    root_dir = _resolve_path(root_dir)
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Not a directory: {root_dir}")
    candidates = []
    for name in os.listdir(root_dir):
        if not name.startswith("eval_outputs_"):
            continue
        p = os.path.join(root_dir, name, "trajectory_metrics_summary.json")
        if os.path.exists(p):
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError(
            f"No trajectory_metrics_summary.json found under {root_dir}/eval_outputs_*"
        )
    # Directory name includes timestamp; lexicographic sort works.
    return sorted(candidates)[-1]


def _get_score(block: Dict[str, Any], model_name: str, auc_degs: List[int]) -> Score:
    models = block.get("models", {})
    m = models.get(model_name)
    if not isinstance(m, dict):
        return Score(None, {}, {})
    ate_mean = None
    try:
        ate_mean = m.get("ate_rmse", {}).get("mean", None)
    except Exception:
        ate_mean = None

    auc: Dict[str, Optional[float]] = {}
    auc_pct: Dict[str, Optional[float]] = {}
    for deg in auc_degs:
        auc_key = f"auc_{int(deg)}deg"
        v = None
        try:
            v = m.get("auc", {}).get(auc_key, None)
        except Exception:
            v = None
        auc[auc_key] = None if v is None else float(v)
        auc_pct[auc_key] = None if v is None else float(v) * 100.0

    return Score(
        None if ate_mean is None else float(ate_mean),
        auc,
        auc_pct,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract final ATE(mean) and AUC from trajectory_metrics_summary.json"
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to trajectory_metrics_summary.json (default: latest under Comparative_tests/).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for final_scores.json (default: alongside input).",
    )
    parser.add_argument(
        "--auc-deg",
        nargs="+",
        type=int,
        default=[10],
        help="One or more AUC thresholds in degrees (e.g., --auc-deg 5 10). Default: 10.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["VGGT", "Any4D", "ViPE", "MapAnything", "DA3"],
        help="Model names to extract (order preserved).",
    )
    args = parser.parse_args()

    if args.input is None:
        input_path = _find_latest_summary_json(SCRIPT_DIR)
    else:
        input_path = _resolve_path(args.input)
    data = _load_json(input_path)

    if args.output is None:
        out_dir = os.path.dirname(input_path)
        out_path = os.path.join(out_dir, "final_scores.json")
    else:
        out_path = _resolve_path(args.output)
        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)

    up = data.get("up_to_scale", {})
    metric = data.get("metric", {})

    # Compact dict for paper tables.
    out: Dict[str, Any] = {
        "source_summary": input_path,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "auc_deg": [int(x) for x in args.auc_deg],
        "up_to_scale": {"models": {}},
        "metric": {"models": {}},
        "table_rows": [],
    }

    for model in args.models:
        s_up = _get_score(up, model, args.auc_deg)
        s_m = _get_score(metric, model, args.auc_deg)

        out["up_to_scale"]["models"][model] = {
            "ate_mean": s_up.ate_mean,
            "auc": s_up.auc,
            "auc_pct": s_up.auc_pct,
        }
        out["metric"]["models"][model] = {
            "ate_mean": s_m.ate_mean,
            "auc": s_m.auc,
            "auc_pct": s_m.auc_pct,
        }

        primary_deg = max(int(x) for x in args.auc_deg) if args.auc_deg else 10
        primary_key = f"auc_{primary_deg}deg"

        out["table_rows"].append(
            {
                "method": model,
                "metric_supported": (model != "VGGT"),
                "up_to_scale_ate_mean": s_up.ate_mean,
                "up_to_scale_auc_pct": s_up.auc_pct.get(primary_key),
                "metric_ate_mean": s_m.ate_mean,
                "metric_auc_pct": s_m.auc_pct.get(primary_key),
            }
        )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"✓ input:  {input_path}")
    print(f"✓ output: {out_path}")


if __name__ == "__main__":
    main()

