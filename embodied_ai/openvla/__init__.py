from pathlib import Path

from .adapter import OpenVLAAdapter, OpenVLAAdapterConfig

ENCODER_NAME = 'openvla'

def build_adapter(local_repo_root: str | None = None):
    del local_repo_root
    cfg = OpenVLAAdapterConfig(encoder_name=ENCODER_NAME)
    embodied_root = Path(__file__).resolve().parents[1]
    cfg.dino_model_id = str(embodied_root / "assets/dinov2/facebook--dinov2-base")
    cfg.siglip_model_id = str(embodied_root / "assets/siglip/google--siglip-base-patch16-224")
    return OpenVLAAdapter(cfg)
