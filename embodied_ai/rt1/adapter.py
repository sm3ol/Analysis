"""RT-1 adapter using the real local RT1 image tokenizer backbone."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from embodied_ai.common.types import AdapterOutput, TrainBatch

from embodied_ai.common.base import AdapterConfig, EncoderAdapter


@dataclass
class RT1AdapterConfig(AdapterConfig):
    """RT-1 adapter settings."""

    checkpoint_path: str | None = None
    use_token_learner: bool = True
    num_tokens: int = 8
    image_size: int = 300
    output_dim: int = 256
    local_repo_root: str = "vendor/rt1_pytorch"


class RT1Adapter(EncoderAdapter):
    """Extracts RT-1 token embeddings in framework format."""

    def __init__(self, config: RT1AdapterConfig):
        super().__init__(config=config)
        rt1_root = self._resolve_local_repo_root(config.local_repo_root)
        if not rt1_root.exists():
            raise RuntimeError(
                f"RT1Adapter expected local repo at `{rt1_root}`. "
                "Set RT1AdapterConfig.local_repo_root to the rt1_pytorch path."
            )
        if str(rt1_root) not in sys.path:
            sys.path.insert(0, str(rt1_root))

        try:
            from maruya24_rt1.tokenizers.image_tokenizer import RT1ImageTokenizer
        except Exception as e:
            raise RuntimeError(
                "RT1Adapter failed to import maruya24_rt1 tokenizer. "
                "Ensure rt1_pytorch dependencies are installed."
            ) from e

        self.tokenizer = RT1ImageTokenizer(
            embedding_output_dim=512,
            language_embedding_size=512,
            use_token_learner=bool(config.use_token_learner),
            num_tokens=int(config.num_tokens),
        )
        self.tokenizer.eval()
        self.tokenizer.requires_grad_(not bool(config.freeze_encoder))

        self.register_buffer(
            "_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )

    @staticmethod
    def _resolve_local_repo_root(local_repo_root: str) -> Path:
        path = Path(local_repo_root)
        if path.is_absolute():
            return path
        embodied_root = Path(__file__).resolve().parents[1]
        candidate = embodied_root / path
        if candidate.exists():
            return candidate
        return path

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
        if x.ndim == 4:
            x = x.unsqueeze(1)  # [B,1,C,H,W]
        if x.ndim != 5:
            raise ValueError(f"RT1Adapter expects images [B,C,H,W] or [B,T,C,H,W], got {x.shape}")

        b, t, c, h, w = x.shape
        x = x.float().clamp(0.0, 1.0)
        size = int(self.config.image_size)
        if h != size or w != size:
            flat = x.view(b * t, c, h, w)
            flat = F.interpolate(flat, size=(size, size), mode="bilinear", align_corners=False)
            x = flat.view(b, t, c, size, size)
        x = (x - self._mean) / self._std

        context = torch.zeros((b, t, 512), device=x.device, dtype=x.dtype)
        tok = self.tokenizer(x, context=context)  # [B,T,N,512]
        if tok.ndim != 4:
            raise RuntimeError(f"RT1 tokenizer returned unexpected shape: {tok.shape}")
        tokens = tok[:, -1, :, :]  # keep latest timestep -> [B,N,512]
        tokens = self._match_dim(tokens, int(self.config.output_dim))

        mask = torch.ones((b, tokens.shape[1]), dtype=torch.bool, device=tokens.device)
        return AdapterOutput(tokens=tokens, token_mask=mask, extras={"backbone": "rt1_image_tokenizer"})
