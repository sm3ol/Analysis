"""Shared helpers for loading local embodied analysis sample episodes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .types import TrainBatch


@dataclass(frozen=True)
class LoadedEpisode:
    """Normalized episode payload loaded from the local analysis dataset."""

    path: Path
    episode_id: int
    frames: torch.Tensor
    source_dataset: str
    original_length: int

    @property
    def num_frames(self) -> int:
        return int(self.frames.shape[0])


def embodied_root() -> Path:
    """Return the embodied analysis root directory."""
    return Path(__file__).resolve().parents[1]


def default_dataset_root() -> Path:
    """Return the default local dataset root for embodied analysis."""
    return embodied_root() / "dataset"


def default_outputs_root() -> Path:
    """Return the default outputs root for embodied analysis."""
    return embodied_root() / "outputs"


def _resolve_root(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def resolve_episode_path(dataset_root: str | Path | None = None, episode_path: str | Path | None = None) -> Path:
    """Resolve the episode file path to use for real-data checks."""
    if episode_path:
        path = _resolve_root(episode_path)
        if not path.exists():
            raise FileNotFoundError(f"episode file not found: {path}")
        return path

    root = default_dataset_root() if dataset_root in (None, "") else _resolve_root(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root not found: {root}")

    candidates = sorted(root.glob("episode_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no episode_*.pt files found in dataset root: {root}")
    return candidates[0]


def load_episode(dataset_root: str | Path | None = None, episode_path: str | Path | None = None) -> LoadedEpisode:
    """Load and validate one local sample episode."""
    path = resolve_episode_path(dataset_root=dataset_root, episode_path=episode_path)
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"expected dict payload in {path}, got {type(payload).__name__}")

    frames = payload.get("frames")
    if not torch.is_tensor(frames):
        raise TypeError(f"expected tensor `frames` in {path}, got {type(frames).__name__}")
    if frames.ndim != 4:
        raise ValueError(f"`frames` must be [T,H,W,C], got {tuple(frames.shape)}")
    if int(frames.shape[-1]) != 3:
        raise ValueError(f"`frames` last dim must be RGB=3, got {tuple(frames.shape)}")

    return LoadedEpisode(
        path=path,
        episode_id=int(payload.get("episode_id", 0)),
        frames=frames.contiguous(),
        source_dataset=str(payload.get("source_dataset", "unknown")),
        original_length=int(payload.get("original_length", int(frames.shape[0]))),
    )


def select_step_indices(total_frames: int, num_steps: int, min_step: int = 0) -> list[int]:
    """Choose evenly spaced frame indices across an episode."""
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")

    start = max(0, min(int(min_step), total_frames - 1))
    count = max(1, int(num_steps))
    span = total_frames - start
    if span <= 1:
        return [total_frames - 1]
    if count == 1:
        return [total_frames - 1]
    if count >= span:
        return list(range(start, total_frames))

    gap = float(span - 1) / float(count - 1)
    indices: list[int] = []
    for i in range(count):
        idx = start + int(round(i * gap))
        idx = min(total_frames - 1, max(start, idx))
        if not indices or idx != indices[-1]:
            indices.append(idx)
    return indices


def make_batch_for_step(
    episode: LoadedEpisode,
    step_index: int,
    sequence_length: int,
    device: torch.device,
) -> TrainBatch:
    """Create a single-batch window ending at the given frame index."""
    total_frames = int(episode.frames.shape[0])
    if total_frames <= 0:
        raise ValueError("episode contains no frames")

    seq_len = max(1, int(sequence_length))
    end = min(total_frames - 1, max(0, int(step_index)))
    start = max(0, end - seq_len + 1)
    window = episode.frames[start : end + 1]
    if int(window.shape[0]) < seq_len:
        pad_count = seq_len - int(window.shape[0])
        pad = window[:1].repeat(pad_count, 1, 1, 1)
        window = torch.cat([pad, window], dim=0)

    images = window.permute(0, 3, 1, 2).contiguous().float() / 255.0
    images = images.unsqueeze(0).to(device)
    step_value = torch.tensor([end], dtype=torch.long, device=device)

    return TrainBatch(
        images=images,
        episode_id=torch.tensor([int(episode.episode_id)], dtype=torch.long, device=device),
        timestep=step_value,
        corruption_family_id=torch.zeros((1,), dtype=torch.long, device=device),
        is_corrupt=torch.zeros((1,), dtype=torch.long, device=device),
        metadata={
            "episode_path": str(episode.path),
            "source_dataset": episode.source_dataset,
            "window_start": int(start),
            "window_stop": int(end),
            "sequence_length": int(seq_len),
        },
    )


def episode_summary(episode: LoadedEpisode) -> dict[str, int | str | list[int]]:
    """Return a compact JSON-safe summary of a loaded episode."""
    return {
        "episode_path": str(episode.path),
        "episode_id": int(episode.episode_id),
        "source_dataset": episode.source_dataset,
        "original_length": int(episode.original_length),
        "num_frames": int(episode.num_frames),
        "frame_shape": [int(v) for v in episode.frames.shape],
    }
