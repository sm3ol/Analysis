# Embodied Sample Dataset

This folder holds the local real sample episodes used by the embodied analysis
sub-repo.

Current sample:
- `episode_000001.pt`

This sample was copied from the training-side exported Bridge/OpenX episodes so
the analysis scripts can run against a stable local input without depending on
the larger training dataset tree at runtime.

By default, the real-data analysis scripts use the first `episode_*.pt` file in
this folder.
