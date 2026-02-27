#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import torch

from embodied_ai.common.device import resolve_device
from embodied_ai.common.types import TrainBatch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generic adapter self-test.")
    p.add_argument("--module", type=str, required=True, help="Python module path, e.g. embodied_ai.dinov2")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--save_path", type=str, required=True)
    p.add_argument("--local_repo_root", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mod = importlib.import_module(args.module)
    device = resolve_device(args.device)
    adapter = mod.build_adapter(local_repo_root=args.local_repo_root or None).to(device)
    adapter.eval()
    batch = TrainBatch(
        images=torch.rand((int(args.batch_size), 3, int(args.image_size), int(args.image_size)), device=device),
        episode_id=torch.arange(int(args.batch_size), dtype=torch.long, device=device),
        timestep=torch.zeros((int(args.batch_size),), dtype=torch.long, device=device),
        corruption_family_id=torch.zeros((int(args.batch_size),), dtype=torch.long, device=device),
        is_corrupt=torch.zeros((int(args.batch_size),), dtype=torch.long, device=device),
    )
    out: dict[str, Any] = {"module": args.module, "device": str(device), "result": {}}
    try:
        with torch.no_grad():
            res = adapter(batch)
        if res.tokens.ndim != 3:
            raise RuntimeError(f"tokens must be [B,N,C], got {res.tokens.shape}")
        if res.token_mask is None or res.token_mask.shape != res.tokens.shape[:2]:
            raise RuntimeError("token_mask missing or wrong shape")
        out["result"] = {"status": "passed", "tokens_shape": list(res.tokens.shape)}
        print(f"[ADAPTER-TEST] {args.module} passed tokens={tuple(res.tokens.shape)}", flush=True)
    except Exception as e:
        out["result"] = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_path).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[ADAPTER-TEST] {args.module} failed {out['result']['error']}", flush=True)
        raise SystemExit(1)

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save_path).write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
