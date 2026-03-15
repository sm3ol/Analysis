"""Streaming score calibration utilities for AV reliability outputs.

Default mapping:
    drop_t = max(0, baseline - raw_score_t)                       # simple mode
    drop_t = max(0, (baseline - raw_score_t)/(prefix_std + eps))  # normalized mode
    calibrated_t = clamp(min + (max-min) * exp(-alpha * drop_t**gamma), [min, max])

Baseline policy currently supported:
    - fixed_prefix_mean: baseline/std are frozen after the first ``prefix_len`` frames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable


DropMapper = Callable[[float, float, float, float], float]


def exponential_drop_mapper(
    drop: float,
    alpha: float,
    min_score: float,
    max_score: float,
) -> float:
    """Map score drop to calibrated score using a smooth monotonic exponential."""
    return min_score + (max_score - min_score) * math.exp(-alpha * max(0.0, float(drop)))


@dataclass
class OnlineCalibrationConfig:
    """Configuration for streaming score calibration."""

    prefix_len: int = 50
    mode: str = "simple"  # simple | normalized
    alpha: float = 10.0
    gamma: float = 1.0
    eps: float = 1e-6
    min_score: float = 0.05
    max_score: float = 1.0
    baseline_policy: str = "fixed_prefix_mean"
    emit_before_ready: bool = True
    # Optional monotonic two-slope drop shaping (keeps mapping smooth/monotonic).
    use_piecewise_drop: bool = False
    piecewise_knee: float = 0.1
    piecewise_tail_mult: float = 3.0


class OnlineScoreCalibrator:
    """Streaming calibrator for per-frame raw reliability scores.

    Notes:
    - Baseline is computed from the first ``prefix_len`` frames and then frozen.
    - Before baseline is frozen, this class can still emit values online by using
      a running mean of seen prefix scores (``emit_before_ready=True``).
    - To replace exponential mapping later (e.g., sigmoid/piecewise), pass a
      custom ``mapper`` callable.
    """

    def __init__(
        self,
        config: OnlineCalibrationConfig | None = None,
        mapper: DropMapper | None = None,
    ) -> None:
        self.config = OnlineCalibrationConfig() if config is None else config
        self.mapper = exponential_drop_mapper if mapper is None else mapper
        self._prefix_scores: list[float] = []
        self._baseline: float | None = None
        self._baseline_std: float | None = None

        if int(self.config.prefix_len) <= 0:
            raise ValueError("prefix_len must be > 0")
        if str(self.config.mode) not in ("simple", "normalized"):
            raise ValueError("mode must be 'simple' or 'normalized'")
        if float(self.config.alpha) < 0.0:
            raise ValueError("alpha must be >= 0")
        if float(self.config.gamma) <= 0.0:
            raise ValueError("gamma must be > 0")
        if float(self.config.eps) <= 0.0:
            raise ValueError("eps must be > 0")
        if float(self.config.max_score) <= float(self.config.min_score):
            raise ValueError("max_score must be > min_score")
        if str(self.config.baseline_policy) != "fixed_prefix_mean":
            raise ValueError("unsupported baseline_policy; use 'fixed_prefix_mean'")
        if float(self.config.piecewise_knee) < 0.0:
            raise ValueError("piecewise_knee must be >= 0")
        if float(self.config.piecewise_tail_mult) < 1.0:
            raise ValueError("piecewise_tail_mult must be >= 1.0")

    @property
    def baseline(self) -> float | None:
        """Frozen baseline if available, else None."""
        return self._baseline

    @property
    def baseline_std(self) -> float | None:
        """Frozen baseline std if available, else None."""
        return self._baseline_std

    @property
    def ready(self) -> bool:
        """True once the baseline is frozen from prefix frames."""
        return self._baseline is not None

    def reset(self) -> None:
        """Reset internal streaming state."""
        self._prefix_scores.clear()
        self._baseline = None
        self._baseline_std = None

    @staticmethod
    def _std(values: list[float]) -> float:
        if not values:
            return 0.0
        mu = float(sum(values) / len(values))
        var = sum((float(v) - mu) ** 2 for v in values) / float(len(values))
        return float(math.sqrt(var))

    def _shape_drop(self, drop: float) -> float:
        base_drop = max(0.0, float(drop))
        gamma = float(self.config.gamma)
        if not bool(self.config.use_piecewise_drop):
            return base_drop**gamma
        knee = max(0.0, float(self.config.piecewise_knee))
        tail_mult = max(1.0, float(self.config.piecewise_tail_mult))
        head = min(base_drop, knee)
        tail = max(0.0, base_drop - knee)
        return (head**gamma) + tail_mult * (tail**gamma)

    def _compute_drop(self, baseline: float, baseline_std: float, raw_score: float) -> float:
        delta = float(baseline) - float(raw_score)
        drop = max(0.0, delta)
        if str(self.config.mode) == "normalized":
            return drop / (float(baseline_std) + float(self.config.eps))
        return drop

    def observe(self, raw_score: float) -> float:
        """Consume one raw score and return one calibrated score."""
        x = float(raw_score)
        if self._baseline is None:
            self._prefix_scores.append(x)
            if len(self._prefix_scores) >= int(self.config.prefix_len):
                prefix = self._prefix_scores[: int(self.config.prefix_len)]
                self._baseline = float(sum(prefix) / len(prefix))
                self._baseline_std = self._std(prefix)
                baseline = self._baseline
                baseline_std = float(self._baseline_std or 0.0)
            else:
                if not bool(self.config.emit_before_ready):
                    return max(float(self.config.min_score), min(float(self.config.max_score), x))
                baseline = float(sum(self._prefix_scores) / len(self._prefix_scores))
                baseline_std = self._std(self._prefix_scores)
        else:
            baseline = self._baseline
            baseline_std = float(self._baseline_std or 0.0)

        drop = self._compute_drop(baseline=baseline, baseline_std=baseline_std, raw_score=x)
        shaped_drop = self._shape_drop(drop)
        mapped = float(
            self.mapper(
                shaped_drop,
                float(self.config.alpha),
                float(self.config.min_score),
                float(self.config.max_score),
            )
        )
        return max(float(self.config.min_score), min(float(self.config.max_score), mapped))

    def calibrate_stream(self, raw_scores: Iterable[float]) -> list[float]:
        """Calibrate a full stream while preserving streaming behavior."""
        return [self.observe(v) for v in raw_scores]


def example_usage() -> dict[str, list[float] | float]:
    """Small example on a single stream of raw scores."""
    raw_stream = [0.52, 0.53, 0.51, 0.52, 0.50, 0.48, 0.35, 0.31, 0.49, 0.51]
    cfg = OnlineCalibrationConfig(prefix_len=5, mode="simple", alpha=10.0, gamma=1.0, min_score=0.05, max_score=1.0)
    calibrator = OnlineScoreCalibrator(config=cfg)
    calibrated = calibrator.calibrate_stream(raw_stream)
    return {
        "raw_stream": raw_stream,
        "calibrated_stream": calibrated,
        "frozen_baseline": float(calibrator.baseline if calibrator.baseline is not None else float("nan")),
    }
