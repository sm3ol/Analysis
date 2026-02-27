# dinov2 Encoder Pack

This folder contains the real dinov2 adapter plus ready-to-run checks.

Files:
- adapter.py: real adapter implementation
- run_encoder_only.sh: adapter-only self-test
- run_with_scorer.sh: encoder + scorer runtime self-test
- smoke_encoder_only.sh: lightweight wrapper for adapter-only check
- smoke_with_scorer.sh: lightweight wrapper for encoder + scorer check

## Run

From this folder:

bash run_encoder_only.sh
bash run_with_scorer.sh

DINOv2 note:
- By default this pack uses `../assets/dinov2/facebook--dinov2-base`.
- You can override it with: `--local_repo_root /path/to/facebook--dinov2-base`
