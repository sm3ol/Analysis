#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from embodied_ai.common.corruptions import CORRUPTION_CHOICES, apply_corruption_frames
from embodied_ai.common.dataset import default_outputs_root, episode_summary, load_episode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a corrupted copy of the local sample episode.")
    p.add_argument("--dataset_root", type=str, default="")
    p.add_argument("--episode_path", type=str, default="")
    p.add_argument("--corruption", type=str, default="noise_and_occlusion", choices=CORRUPTION_CHOICES)
    p.add_argument("--severity", type=float, default=0.8)
    p.add_argument("--start_step", type=int, default=2)
    p.add_argument("--end_step", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_path", type=str, default="")
    return p.parse_args()


def _default_output_path(episode_id: int, corruption: str) -> Path:
    out_dir = default_outputs_root() / "corrupted"
    return out_dir / f"episode_{episode_id:06d}_{corruption}.pt"


def main() -> None:
    args = parse_args()
    episode = load_episode(dataset_root=args.dataset_root or None, episode_path=args.episode_path or None)

    start = max(0, min(int(args.start_step), episode.num_frames - 1))
    end = max(start, min(int(args.end_step), episode.num_frames - 1))
    frames = episode.frames.clone()
    frames[start : end + 1] = apply_corruption_frames(
        frames[start : end + 1],
        kind=args.corruption,
        severity=float(args.severity),
        seed=int(args.seed),
    )

    out_path = Path(args.output_path) if args.output_path else _default_output_path(episode.episode_id, args.corruption)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "episode_id": int(episode.episode_id),
        "frames": frames,
        "source_dataset": episode.source_dataset,
        "original_length": int(episode.original_length),
        "source_episode_path": str(episode.path),
        "corruption": args.corruption,
        "corruption_severity": float(args.severity),
        "corruption_range": [int(start), int(end)],
    }
    torch.save(payload, out_path)

    summary = {
        "status": "passed",
        "input_episode": episode_summary(episode),
        "output_path": str(out_path),
        "corruption": args.corruption,
        "severity": float(args.severity),
        "corruption_range": [int(start), int(end)],
    }
    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[CORRUPTION] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
