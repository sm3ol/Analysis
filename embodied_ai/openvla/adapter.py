"""OpenVLA adapter with real fused vision backbones (DINOv2 + SigLIP)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from embodied_ai.common.types import AdapterOutput, TrainBatch

from embodied_ai.common.base import AdapterConfig, EncoderAdapter


@dataclass
class OpenVLAAdapterConfig(AdapterConfig):
    """OpenVLA adapter settings."""

    model_id: str = "openvla/openvla-7b"
    dino_model_id: str = "assets/dinov2/facebook--dinov2-base"
    siglip_model_id: str = "assets/siglip/google--siglip-base-patch16-224"
    local_files_only: bool = True
    image_size: int = 224
    output_dim: int = 256
    use_global_fused_token: bool = True


class OpenVLAAdapter(EncoderAdapter):
    """Extracts OpenVLA-style fused vision embeddings in framework format."""

    def __init__(self, config: OpenVLAAdapterConfig):
        super().__init__(config=config)
        try:
            from transformers import AutoModel, SiglipVisionModel
        except Exception as e:
            raise RuntimeError(
                "OpenVLAAdapter requires `transformers`. Install with `pip install transformers`."
            ) from e

        # OpenVLA's vision stack is built on fused DINO/SigLIP families.
        # We use the real pretrained backbones and fuse their global embeddings.
        self.dino = AutoModel.from_pretrained(
            self._resolve_model_ref(config.dino_model_id),
            local_files_only=bool(config.local_files_only),
        )
        self.siglip = SiglipVisionModel.from_pretrained(
            self._resolve_model_ref(config.siglip_model_id),
            local_files_only=bool(config.local_files_only),
        )
        self.dino.eval()
        self.siglip.eval()
        self.dino.requires_grad_(not bool(config.freeze_encoder))
        self.siglip.requires_grad_(not bool(config.freeze_encoder))

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
    def _resolve_model_ref(model_ref: str) -> str:
        path = Path(model_ref)
        if path.is_absolute():
            return str(path)
        embodied_root = Path(__file__).resolve().parents[1]
        candidate = embodied_root / path
        if candidate.exists():
            return str(candidate)
        return model_ref

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
            raise ValueError(f"OpenVLAAdapter expects images [B,C,H,W] or [B,T,C,H,W], got {x.shape}")

        x = x.float().clamp(0.0, 1.0)
        size = int(self.config.image_size)
        if x.shape[-1] != size or x.shape[-2] != size:
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        x = (x - self._mean) / self._std

        dino_out = self.dino(pixel_values=x).last_hidden_state
        siglip_out = self.siglip(pixel_values=x).last_hidden_state
        if dino_out.ndim != 3 or siglip_out.ndim != 3:
            raise RuntimeError(
                f"OpenVLA fused backbones returned unexpected shapes: dino={dino_out.shape}, siglip={siglip_out.shape}"
            )

        if bool(self.config.use_global_fused_token):
            dino_global = dino_out[:, 0, :]
            siglip_global = siglip_out[:, 0, :]
            fused = torch.cat([dino_global, siglip_global], dim=-1)
            tokens = fused.unsqueeze(1)
        else:
            dino_tokens = dino_out[:, 1:, :] if dino_out.shape[1] > 1 else dino_out
            siglip_tokens = siglip_out[:, 1:, :] if siglip_out.shape[1] > 1 else siglip_out
            n = min(dino_tokens.shape[1], siglip_tokens.shape[1])
            tokens = torch.cat([dino_tokens[:, :n, :], siglip_tokens[:, :n, :]], dim=-1)

        tokens = self._match_dim(tokens, int(self.config.output_dim))
        b = tokens.shape[0]
        mask = torch.ones((b, tokens.shape[1]), dtype=torch.bool, device=tokens.device)
        return AdapterOutput(tokens=tokens, token_mask=mask, extras={"backbone": "openvla_fused_dino_siglip"})
