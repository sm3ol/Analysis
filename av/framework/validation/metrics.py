"""Validation metrics for temporal reliability and recovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import ValidationConfig


@dataclass
class EpisodeTrace:
    """Collected per-episode validation trace."""

    episode_id: int
    split: str
    corruption_length: int
    corrupt_start: int
    corrupt_end: int
    alarm: list[int]
    mode_name: list[str]
    final_reliability: list[float]
    is_corrupt: list[int]


def _binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    scores = np.concatenate([pos, neg], axis=0)
    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    idx_pos = np.where(y_true == 1)[0]
    r_pos = ranks[idx_pos]
    u = np.sum(r_pos) - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def _length_bucket(length: int, cfg: ValidationConfig) -> str:
    if length <= int(cfg.short_max_len):
        return "shorter"
    if length <= int(cfg.medium_max_len):
        return "medium"
    return "longer"


def _first_true(mask: np.ndarray) -> int | None:
    idx = np.where(mask > 0)[0]
    if len(idx) == 0:
        return None
    return int(idx[0])


def _aggregate_traces(traces: list[EpisodeTrace], include_auroc: bool) -> dict[str, float]:
    if not traces:
        return {
            "episode_count": 0.0,
            "prefix_false_alarm_rate": float("nan"),
            "corrupt_alarm_rate": float("nan"),
            "recovery_false_alarm_rate": float("nan"),
            "time_to_detect": float("nan"),
            "time_to_recover": float("nan"),
            "frame_auroc": float("nan") if include_auroc else float("nan"),
        }

    prefix_fars = []
    corrupt_alarm_rates = []
    recovery_fars = []
    detect_times = []
    recover_times = []
    auroc_true = []
    auroc_score = []

    for trace in traces:
        alarm = np.asarray(trace.alarm, dtype=np.int64)
        is_corrupt = np.asarray(trace.is_corrupt, dtype=np.int64)
        reliability = np.asarray(trace.final_reliability, dtype=np.float32)

        prefix = alarm[: trace.corrupt_start]
        corrupt = alarm[trace.corrupt_start : trace.corrupt_end]
        recovery = alarm[trace.corrupt_end :]

        prefix_fars.append(float(prefix.mean()) if prefix.size else 0.0)
        corrupt_alarm_rates.append(float(corrupt.mean()) if corrupt.size else 0.0)
        recovery_fars.append(float(recovery.mean()) if recovery.size else 0.0)

        det_idx = _first_true(corrupt)
        detect_times.append(float(len(corrupt) if det_idx is None else det_idx))

        clean_tail = (recovery == 0).astype(np.int64)
        rec_idx = _first_true(clean_tail)
        recover_times.append(float(len(recovery) if rec_idx is None else rec_idx))

        auroc_true.append(is_corrupt)
        auroc_score.append(1.0 - reliability)

    metrics = {
        "episode_count": float(len(traces)),
        "prefix_false_alarm_rate": float(np.mean(prefix_fars)),
        "corrupt_alarm_rate": float(np.mean(corrupt_alarm_rates)),
        "recovery_false_alarm_rate": float(np.mean(recovery_fars)),
        "time_to_detect": float(np.mean(detect_times)),
        "time_to_recover": float(np.mean(recover_times)),
    }
    if include_auroc:
        y_true = np.concatenate(auroc_true, axis=0)
        y_score = np.concatenate(auroc_score, axis=0)
        metrics["frame_auroc"] = _binary_auroc(y_true, y_score)
    return metrics


def summarize_split(traces: list[EpisodeTrace], cfg: ValidationConfig) -> dict[str, Any]:
    """Summarize temporal metrics overall and by corruption-length bucket."""
    buckets = {"shorter": [], "medium": [], "longer": []}
    for trace in traces:
        buckets[_length_bucket(trace.corruption_length, cfg)].append(trace)

    return {
        "overall": _aggregate_traces(traces, include_auroc=cfg.include_auroc),
        "by_length_bucket": {
            bucket_name: _aggregate_traces(bucket_traces, include_auroc=cfg.include_auroc)
            for bucket_name, bucket_traces in buckets.items()
        },
    }

