# Embodied Sample Dataset

This folder holds the local real sample episode used by the embodied inference
profiling workflow.

Current staged sample:
- `episode_000001.pt`

Notes:
- We intentionally keep exactly one local episode in-repo so inference scripts
  run out of the box.
- The 10-episode source set lives outside this repo at:
  - `/home/dal574571/ASPLOS 27/data/openx_raw/exports_droid_subset_10eps_min250`
- By default, real-data scripts pick the first `episode_*.pt` in this folder.
