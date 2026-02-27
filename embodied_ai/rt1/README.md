# rt1 Encoder Pack

This folder contains the real rt1 adapter plus ready-to-run checks.

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

RT1 note:
- By default this pack uses `../vendor/rt1_pytorch`, which contains a local
  `maruya24_rt1` snapshot.
- You can still override it with: `--local_repo_root /path/to/rt1_pytorch`
