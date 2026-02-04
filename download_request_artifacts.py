#!/usr/bin/env python3
"""
python3 download_request_artifacts.py \
  -u "https://ea5i5e07fh1w-s8tvic3pz6qy.serving.hyperai.host" \
  -r "20260120_184722_58724cc6" \
  --only mp4 \
  --out-dir "./downloads"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable


def _join_url(base: str, path: str) -> str:
    return base.rstrip("/") + "/" + path.lstrip("/")


def _request(url: str, *, accept: str | None = None) -> urllib.request.Request:
    headers = {"User-Agent": "download_request_artifacts/1.0"}
    if accept:
        headers["Accept"] = accept
    return urllib.request.Request(url, headers=headers, method="GET")


def _download_with_retries(
    url: str,
    out_path: Path,
    *,
    accept: str | None,
    timeout_sec: float,
    retries: int,
    backoff_sec: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        if attempt > 0:
            sleep_s = backoff_sec * (2 ** (attempt - 1))
            print(f"[retry] attempt {attempt}/{retries} after {sleep_s:.1f}s: {url}", file=sys.stderr)
            time.sleep(sleep_s)
        try:
            req = _request(url, accept=accept)
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                status = getattr(resp, "status", None)
                if status and status >= 400:
                    raise urllib.error.HTTPError(url, status, "HTTP error", hdrs=resp.headers, fp=None)

                tmp = out_path.with_suffix(out_path.suffix + ".part")
                size = 0
                with open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        size += len(chunk)

                os.replace(tmp, out_path)
                ct = resp.headers.get("Content-Type", "")
                print(f"[ok] {out_path} ({size} bytes)  content-type={ct}")
                return
        except Exception as e:  # noqa: BLE001 - keep it simple for CLI
            last_err = e

    raise RuntimeError(f"Failed to download after {retries + 1} attempts: {url}\nLast error: {last_err}") from last_err


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("-u", "--service-url", required=True, help="Base URL, e.g. https://xxx.serving.hyperai.host")
    p.add_argument("-r", "--request-id", required=True, help="Request id, e.g. 20260120_184722_58724cc6")
    p.add_argument(
        "--out-dir",
        default=".",
        help="Output directory (default: current directory)",
    )
    p.add_argument(
        "--only",
        choices=["all", "json", "mp4"],
        default="all",
        help="Download only json or mp4 (default: all)",
    )
    p.add_argument("--timeout-sec", type=float, default=600.0, help="Per-request timeout seconds (default: 600)")
    p.add_argument("--retries", type=int, default=2, help="Retry times on failure (default: 2)")
    p.add_argument("--backoff-sec", type=float, default=1.0, help="Retry backoff base seconds (default: 1)")
    return p.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = _parse_args(argv)

    out_dir = Path(args.out_dir).expanduser().resolve() / args.request_id
    base = args.service_url
    rid = args.request_id

    if args.only in {"all", "json"}:
        url = _join_url(base, f"/requests/{rid}/result.json")
        out_path = out_dir / "result.json"
        _download_with_retries(
            url,
            out_path,
            accept="application/json",
            timeout_sec=args.timeout_sec,
            retries=args.retries,
            backoff_sec=args.backoff_sec,
        )
        # quick sanity check: is it valid JSON?
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            got = payload.get("request_id")
            if got and got != rid:
                print(f"[warn] result.json request_id mismatch: {got} != {rid}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] result.json is not valid UTF-8/JSON: {e}", file=sys.stderr)

    if args.only in {"all", "mp4"}:
        url = _join_url(base, f"/requests/{rid}/wan_output.mp4")
        out_path = out_dir / "wan_output.mp4"
        _download_with_retries(
            url,
            out_path,
            accept="video/mp4",
            timeout_sec=args.timeout_sec,
            retries=args.retries,
            backoff_sec=args.backoff_sec,
        )

    print(f"[done] saved under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

