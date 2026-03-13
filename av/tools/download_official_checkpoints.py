#!/usr/bin/env python3
"""Download official AV encoder checkpoints into the standalone AV pack."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

AV_ROOT = Path(__file__).resolve().parents[1]
if str(AV_ROOT) not in sys.path:
    sys.path.insert(0, str(AV_ROOT))

from framework.adapters.checkpoint_catalog import (  # noqa: E402
    OFFICIAL_CHECKPOINTS,
    canonical_encoder_name,
    default_checkpoint_path,
)

DEFAULT_ENCODERS = ["pointpillars", "pointrcnn", "pvrcnn", "centerpoint"]


def parse_encoder_list(raw: str) -> list[str]:
    text = str(raw).strip().lower()
    if not text or text == "all":
        return list(DEFAULT_ENCODERS)
    out = [canonical_encoder_name(x.strip()) for x in text.split(",") if x.strip()]
    if not out:
        raise ValueError("encoders list is empty")
    return out


def _download_one(file_id: str, output_path: Path) -> tuple[bool, str]:
    cmd = ["gdown", f"https://drive.google.com/uc?id={file_id}", "-O", str(output_path)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
        return True, ""
    error = proc.stderr.strip() or proc.stdout.strip() or f"gdown failed with code {proc.returncode}"
    return False, error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download official AV encoder checkpoints.")
    parser.add_argument("--encoders", type=str, default="all")
    parser.add_argument("--checkpoint_root", type=str, default="")
    parser.add_argument("--skip_existing", type=int, choices=[0, 1], default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = parse_encoder_list(args.encoders)
    checkpoint_root = Path(args.checkpoint_root).expanduser() if args.checkpoint_root else None
    if checkpoint_root is not None:
        checkpoint_root.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, object]] = {}
    final_root = None
    for encoder_name in requested:
        if encoder_name not in OFFICIAL_CHECKPOINTS:
            results[encoder_name] = {
                "requested": True,
                "downloaded": False,
                "error": f"no catalog entry for encoder '{encoder_name}'",
            }
            continue
        entry = OFFICIAL_CHECKPOINTS[encoder_name]
        output_path = default_checkpoint_path(encoder_name, checkpoint_root=checkpoint_root)
        final_root = output_path.parent
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if bool(int(args.skip_existing)) and output_path.exists() and output_path.stat().st_size > 0:
            results[encoder_name] = {
                "requested": True,
                "downloaded": True,
                "path": str(output_path),
                "source_url": entry.source_url,
                "source_repo": entry.source_repo,
                "skipped_existing": True,
            }
            continue

        ok, error = _download_one(entry.file_id, output_path)
        results[encoder_name] = {
            "requested": True,
            "downloaded": bool(ok),
            "path": str(output_path),
            "source_url": entry.source_url,
            "source_repo": entry.source_repo,
            "skipped_existing": False,
            "error": error or None,
        }

    print(json.dumps({"checkpoint_root": str(final_root) if final_root is not None else "", "results": results}, indent=2))


if __name__ == "__main__":
    main()
