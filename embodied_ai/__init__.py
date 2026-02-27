"""Embodied AI inference analysis package."""

from .common.import_fixes import preload_external_av

try:
    preload_external_av()
except Exception:
    # Best-effort guard against the sibling Analysis/av package shadowing PyAV.
    # If PyAV is not installed, the encoder that needs it will still fail later
    # with its own dependency error.
    pass
