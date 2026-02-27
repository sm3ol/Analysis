#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import torch

from embodied_ai.common.device import resolve_device
from embodied_ai.common.dataset import episode_summary, load_episode, make_batch_for_step


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one encoder on the local real sample episode.")
    p.add_argument("--module", type=str, required=True, help="Python module path, e.g. embodied_ai.dinov2")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dataset_root", type=str, default="")
    p.add_argument("--episode_path", type=str, default="")
    p.add_argument("--sequence_length", type=int, default=4)
    p.add_argument("--step_index", type=int, default=-1, help="Frame index to score; default is the final frame.")
    p.add_argument("--save_path", type=str, required=True)
    p.add_argument("--local_repo_root", type=str, default="")
    return p.parse_args()


def _write_result(path: str, payload: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out: dict[str, Any] = {"module": args.module, "device": args.device, "result": {}}

    try:
        episode = load_episode(dataset_root=args.dataset_root or None, episode_path=args.episode_path or None)
        device = resolve_device(args.device)
        step_index = int(episode.num_frames - 1 if int(args.step_index) < 0 else args.step_index)
        batch = make_batch_for_step(
            episode=episode,
            step_index=step_index,
            sequence_length=int(args.sequence_length),
            device=device,
        )

        mod = importlib.import_module(args.module)
        adapter = mod.build_adapter(local_repo_root=args.local_repo_root or None).to(device)
        adapter.eval()

        with torch.no_grad():
            res = adapter(batch)

        if res.tokens.ndim != 3:
            raise RuntimeError(f"tokens must be [B,N,C], got {tuple(res.tokens.shape)}")
        if res.token_mask is None or tuple(res.token_mask.shape) != tuple(res.tokens.shape[:2]):
            raise RuntimeError("token_mask missing or wrong shape")

        out["result"] = {
            "status": "passed",
            "episode": episode_summary(episode),
            "batch_images_shape": [int(v) for v in batch.images.shape],
            "tokens_shape": [int(v) for v in res.tokens.shape],
            "token_mask_shape": [int(v) for v in res.token_mask.shape],
            "window_start": int(batch.metadata["window_start"]),
            "window_stop": int(batch.metadata["window_stop"]),
        }
        print(
            f"[REAL-ADAPTER] {args.module} passed "
            f"window={batch.metadata['window_start']}:{batch.metadata['window_stop']} "
            f"tokens={tuple(res.tokens.shape)}",
            flush=True,
        )
    except Exception as e:
        out["result"] = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
        _write_result(args.save_path, out)
        print(f"[REAL-ADAPTER] {args.module} failed {out['result']['error']}", flush=True)
        raise SystemExit(1)

    _write_result(args.save_path, out)


if __name__ == "__main__":
    main()
