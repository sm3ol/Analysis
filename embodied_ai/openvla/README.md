# openvla Encoder Pack

This folder contains the real openvla adapter plus ready-to-run checks.

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

OpenVLA note:
- By default this pack uses the local DINOv2 and SigLIP checkpoints under
  `../assets/`.
- It does not need the external Hugging Face cache at runtime anymore.
