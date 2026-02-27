# Embodied Assets

This folder holds local model assets that let `Analysis/embodied_ai` run as a
standalone repo.

Current contents:
- `dinov2/facebook--dinov2-base/`: local DINOv2 checkpoint
- `siglip/google--siglip-base-patch16-224/`: local SigLIP vision checkpoint

These assets are resolved from paths inside `Analysis/embodied_ai` so the smoke
tests do not need to reach back into the parent ASPLOS repo at runtime.
