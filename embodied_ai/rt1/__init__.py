from pathlib import Path

from .adapter import RT1Adapter, RT1AdapterConfig

ENCODER_NAME = 'rt1'

def build_adapter(local_repo_root: str | None = None):
    cfg = RT1AdapterConfig(encoder_name=ENCODER_NAME)
    if local_repo_root:
        cfg.local_repo_root = local_repo_root
    else:
        embodied_root = Path(__file__).resolve().parents[1]
        cfg.local_repo_root = str(embodied_root / "vendor/rt1_pytorch")
    return RT1Adapter(cfg)
