"""SigLIP (HF) adapter using real pretrained vision model."""

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
class SigLIPAdapterConfig(AdapterConfig):
    """SigLIP adapter settings."""

    model_path: str = "assets/siglip/google--siglip-base-patch16-224"
    local_files_only: bool = True
    image_size: int = 224
    output_dim: int = 256
    include_cls_token: bool = False
    max_tokens: Optional[int] = None


class SigLIPAdapter(EncoderAdapter):
    """Extracts SigLIP embeddings in framework format."""

    def __init__(self, config: SigLIPAdapterConfig):
        super().__init__(config=config)
        try:
            from transformers import SiglipVisionModel
        except Exception as e:
            raise RuntimeError(
                "SigLIPAdapter requires `transformers`. Install with `pip install transformers`."
            ) from e

        model_id = self._resolve_model_path(config.model_path)
        self.vision = SiglipVisionModel.from_pretrained(
            model_id,
            local_files_only=bool(config.local_files_only),
        )
        self.vision.eval()
        self.vision.requires_grad_(not bool(config.freeze_encoder))

        self.register_buffer(
            "_mean",
            torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std",
            torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    @staticmethod
    def _resolve_model_path(model_path: str) -> str:
        path = Path(model_path)
        if path.is_absolute():
            return str(path)
        embodied_root = Path(__file__).resolve().parents[1]
        candidate = embodied_root / path
        if candidate.exists():
            return str(candidate)
        return model_path

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
            raise ValueError(f"SigLIPAdapter expects images [B,C,H,W] or [B,T,C,H,W], got {x.shape}")

        x = x.float().clamp(0.0, 1.0)
        size = int(self.config.image_size)
        if x.shape[-1] != size or x.shape[-2] != size:
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        x = (x - self._mean) / self._std

        out = self.vision(pixel_values=x)
        tokens = out.last_hidden_state
        if tokens.ndim != 3:
            raise RuntimeError(f"SigLIP vision model returned unexpected shape: {tokens.shape}")
        if not bool(self.config.include_cls_token) and tokens.shape[1] > 1:
            tokens = tokens[:, 1:, :]
        max_tokens = self.config.max_tokens
        if max_tokens is not None and int(max_tokens) > 0:
            tokens = tokens[:, : int(max_tokens), :]
        tokens = self._match_dim(tokens, int(self.config.output_dim))

        b = tokens.shape[0]
        mask = torch.ones((b, tokens.shape[1]), dtype=torch.bool, device=tokens.device)
        return AdapterOutput(tokens=tokens, token_mask=mask, extras={"backbone": "siglip_hf"})
