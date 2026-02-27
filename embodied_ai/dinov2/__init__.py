from pathlib import Path

from .adapter import DINOv2Adapter, DINOv2AdapterConfig

ENCODER_NAME = 'dinov2'

def build_adapter(local_repo_root: str | None = None):
    del local_repo_root
    cfg = DINOv2AdapterConfig(encoder_name=ENCODER_NAME)
    embodied_root = Path(__file__).resolve().parents[1]
    cfg.model_id = str(embodied_root / "assets/dinov2/facebook--dinov2-base")
    return DINOv2Adapter(cfg)
