"""Local path helpers for the standalone AV inference pack."""

from __future__ import annotations

from pathlib import Path

AV_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = AV_ROOT.parent
CHECKPOINTS_ROOT = AV_ROOT / "checkpoints"
ARTIFACTS_ROOT = AV_ROOT / "artifacts"
OUTPUTS_ROOT = AV_ROOT / "outputs"
VENDOR_ROOT = AV_ROOT / "vendor"
VENDOR_OPENPCDET_ROOT = VENDOR_ROOT / "openpcdet"
OPENPCDET_CFG_ROOT = VENDOR_OPENPCDET_ROOT / "tools" / "cfgs"
