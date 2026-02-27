# Common Inference Modules

This folder contains shared code used by all embodied AI encoder packs.

Included:
- `config.py`
- `types.py`
- `base.py`
- `core/` (pooling, projection, scorer, temporal, Brain-B stats, family stats)
- `runtime.py`
- `scripts/` (generic test runners)

Students should not edit this folder first when profiling a specific encoder.
They should start in the encoder folder (`dinov2/`, `openvla/`, `siglip/`, `rt1/`).
