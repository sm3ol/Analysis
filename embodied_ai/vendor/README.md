# Embodied Vendor Code

This folder holds third-party code snapshots needed by the embodied analysis
sub-repo.

Current contents:
- `rt1_pytorch/`: local RT-1 package snapshot containing `maruya24_rt1`

The RT-1 adapter uses this local copy by default so the embodied smoke tests do
not depend on a sibling checkout outside `Analysis/`.
