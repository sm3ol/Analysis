"""DINOv2 adapter using real HF pretrained backbone."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from embodied_ai.common.types import AdapterOutput, TrainBatch

from embodied_ai.common.base import AdapterConfig, EncoderAdapter


@dataclass
class DINOv2AdapterConfig(AdapterConfig):
    """DINOv2 adapter settings."""

    model_id: str = "assets/dinov2/facebook--dinov2-base"
    image_size: int = 224
    local_files_only: bool = True
    output_dim: int = 256
    include_cls_token: bool = False
    max_tokens: Optional[int] = None


class DINOv2Adapter(EncoderAdapter):
    """Extracts DINOv2 token embeddings in framework format."""

    def __init__(self, config: DINOv2AdapterConfig):
        super().__init__(config=config)
        try:
            from transformers import AutoModel
        except Exception as e:
            raise RuntimeError(
                "DINOv2Adapter requires `transformers`. Install with `pip install transformers`."
            ) from e

        self.backbone = AutoModel.from_pretrained(
            self._resolve_model_id(config.model_id),
            local_files_only=bool(config.local_files_only),
        )
        self.backbone.eval()
        self.backbone.requires_grad_(not bool(config.freeze_encoder))

        self.register_buffer(
            "_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    @staticmethod
    def _resolve_model_id(model_id: str) -> str:
        path = Path(model_id)
        if path.is_absolute():
            return str(path)
        embodied_root = Path(__file__).resolve().parents[1]
        candidate = embodied_root / path
        if candidate.exists():
            return str(candidate)
        return model_id

    @staticmethod
    def _match_dim(tokens: torch.Tensor, target_dim: int) -> torch.Tensor:
        d = int(tokens.shape[-1])
        if d == target_dim:
            return tokens
        if d > target_dim:
            return tokens[..., :target_dim]
        pad = torch.zeros((*tokens.shape[:-1], target_dim - d), device=tokens.device, dtype=tokens.dtype)
        return torch.cat([tokens, pad], dim=-1)

    def encode(self, batch: TrainBatch) -> AdapterOutput:
        x = batch.images
        if x.ndim == 5:
            x = x[:, -1, ...]
        if x.ndim != 4:
            raise ValueError(f"DINOv2Adapter expects images [B,C,H,W] or [B,T,C,H,W], got {x.shape}")

        x = x.float().clamp(0.0, 1.0)
        size = int(self.config.image_size)
        if x.shape[-1] != size or x.shape[-2] != size:
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        x = (x - self._mean) / self._std

        out = self.backbone(pixel_values=x)
        tokens = out.last_hidden_state
        if tokens.ndim != 3:
            raise RuntimeError(f"DINOv2 backbone returned unexpected shape: {tokens.shape}")
        if not bool(self.config.include_cls_token) and tokens.shape[1] > 1:
            tokens = tokens[:, 1:, :]
        max_tokens = self.config.max_tokens
        if max_tokens is not None and int(max_tokens) > 0:
            tokens = tokens[:, : int(max_tokens), :]
        tokens = self._match_dim(tokens, int(self.config.output_dim))

        b = tokens.shape[0]
        mask = torch.ones((b, tokens.shape[1]), dtype=torch.bool, device=tokens.device)
        return AdapterOutput(tokens=tokens, token_mask=mask, extras={"backbone": "dinov2_hf"})
