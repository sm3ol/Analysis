from pathlib import Path

from .adapter import SigLIPAdapter, SigLIPAdapterConfig

ENCODER_NAME = 'siglip'

def build_adapter(local_repo_root: str | None = None):
    cfg = SigLIPAdapterConfig(encoder_name=ENCODER_NAME)
    if local_repo_root:
        cfg.model_path = local_repo_root
    else:
        embodied_root = Path(__file__).resolve().parents[1]
        cfg.model_path = str(embodied_root / "assets/siglip/google--siglip-base-patch16-224")
    return SigLIPAdapter(cfg)
