"""Base class for AV encoder-specific adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ..types import AdapterOutput, TrainBatch


@dataclass
class AdapterConfig:
    """Adapter-specific runtime config."""

    encoder_name: str
    input_dim: int
    hidden_dim: int
    output_dim: int
    freeze_backbone: bool = False
    use_pretrained: bool = False
    checkpoint_path: str | None = None
    strict_checkpoint_load: bool = True
    return_intermediate_shapes: bool = False


class EncoderAdapter(nn.Module, ABC):
    """Abstract adapter that normalizes encoder outputs."""

    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def encode(self, batch: TrainBatch) -> AdapterOutput:
        """Encode input batch into token features and optional masks."""

    def forward(self, batch: TrainBatch) -> AdapterOutput:
        return self.encode(batch)

    def extra_state(self) -> dict[str, Any]:
        return {}

    def freeze_module(self, module: nn.Module) -> None:
        """Freeze parameters in a backbone submodule when configured."""
        if bool(self.config.freeze_backbone):
            module.requires_grad_(False)

    def _extract_state_dict(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            for key in ("state_dict", "model_state", "model_state_dict", "net", "model"):
                candidate = payload.get(key)
                if isinstance(candidate, dict):
                    return candidate
        if not isinstance(payload, dict):
            raise ValueError(
                f"{self.config.encoder_name}: unsupported checkpoint payload type {type(payload).__name__}"
            )
        return payload

    def _normalized_match(
        self,
        module: nn.Module,
        raw_state_dict: dict[str, Any],
        rename_rules: list[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        expected_state = module.state_dict()
        expected = set(expected_state.keys())
        matched: dict[str, Any] = {}
        rename_rules = list(rename_rules or [])
        for key, value in raw_state_dict.items():
            candidates = [key]
            if key.startswith("module."):
                candidates.append(key[len("module.") :])
            if key.startswith("model."):
                candidates.append(key[len("model.") :])
            if key.startswith("state_dict."):
                candidates.append(key[len("state_dict.") :])
            expanded_candidates = list(candidates)
            for candidate in list(candidates):
                for src_prefix, dst_prefix in rename_rules:
                    if candidate.startswith(src_prefix):
                        expanded_candidates.append(dst_prefix + candidate[len(src_prefix) :])
            for candidate in expanded_candidates:
                if candidate in expected and candidate not in matched:
                    target = expected_state[candidate]
                    candidate_value = value
                    if torch.is_tensor(candidate_value) and torch.is_tensor(target):
                        if tuple(candidate_value.shape) != tuple(target.shape):
                            if candidate_value.ndim == 5 and target.ndim == 5:
                                native = candidate_value.transpose(-1, -2).contiguous()
                                if tuple(native.shape) == tuple(target.shape):
                                    candidate_value = native
                                else:
                                    implicit = candidate_value.permute(4, 0, 1, 2, 3).contiguous()
                                    if tuple(implicit.shape) == tuple(target.shape):
                                        candidate_value = implicit
                            if tuple(candidate_value.shape) != tuple(target.shape):
                                continue
                    matched[candidate] = candidate_value
                    break
        return matched

    def load_optional_checkpoint(
        self,
        module: nn.Module,
        rename_rules: list[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Load optional pretrained weights with strict visibility."""
        status = {
            "requested": bool(self.config.use_pretrained),
            "loaded": False,
            "path": self.config.checkpoint_path,
            "strict": bool(self.config.strict_checkpoint_load),
            "matched_keys_count": 0,
            "missing_keys_count": 0,
            "unexpected_keys_count": 0,
            "error": None,
        }
        if not bool(self.config.use_pretrained):
            return status
        if not self.config.checkpoint_path:
            status["error"] = "checkpoint_path is empty while use_pretrained=True"
            return status
        checkpoint_file = str(self.config.checkpoint_path)
        try:
            payload = torch.load(checkpoint_file, map_location="cpu")
            raw_state_dict = self._extract_state_dict(payload)
            matched = self._normalized_match(module, raw_state_dict, rename_rules=rename_rules)
            status["matched_keys_count"] = int(len(matched))
            missing, unexpected = module.load_state_dict(matched, strict=False)
            status["missing_keys_count"] = int(len(missing))
            status["unexpected_keys_count"] = int(len(unexpected))
            strict = bool(self.config.strict_checkpoint_load)
            if strict and (missing or unexpected):
                status["error"] = (
                    f"strict checkpoint load failed: missing={len(missing)}, unexpected={len(unexpected)}"
                )
            else:
                status["loaded"] = True
        except Exception as exc:  # pragma: no cover - exercised in runtime env
            status["error"] = f"{type(exc).__name__}: {exc}"
        return status
