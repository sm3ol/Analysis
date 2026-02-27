# Medical Encoder Subrepo

This folder packages medical encoder paths used by taxonomy scripts.

Included encoders:
- `clip_vit`
- `clip_resnet`
- `biomedclip_vit`
- `medclip_resnet`

Core files:
- `taxonomy_metrics_medmnistc.py`
- `taxonomy_metrics_mediMeta.py`
- `models.py`, `utils.py`
- `run_clip_vit.sh`, `run_clip_resnet.sh`, `run_biomedclip_vit.sh`, `run_medclip_resnet.sh`
- `tools/generate_synthetic_medmnistc.py`
- `smoke_test.sh`

## Environment

Full runtime (real model loading/inference):
```bash
bash setup_env.sh full
```

Smoke only:
```bash
bash setup_env.sh smoke
```

## Usage

Per encoder run (MedMNIST-C style inputs):
```bash
bash run_clip_vit.sh /path/to/medmnist_224 /path/to/medmnistc_generated
bash run_clip_resnet.sh /path/to/medmnist_224 /path/to/medmnistc_generated
bash run_biomedclip_vit.sh /path/to/medmnist_224 /path/to/medmnistc_generated
bash run_medclip_resnet.sh /path/to/medmnist_224 /path/to/medmnistc_generated
```

## Smoke Test (Synthetic Data, No Model Downloads)

```bash
bash smoke_test.sh
```

What it does:
- generates a tiny synthetic MedMNIST-C style dataset under `smoke_data/`
- runs `taxonomy_metrics_medmnistc.py` once per encoder
- uses `--mock_encoder_preflight` to avoid external model downloads
- verifies `metrics_table.csv` exists for each encoder

## Data Requirements

- Full runs need real MedMNIST clean `.npz` files and MedMNIST-C corruption `.npz` files.
- Those datasets are not tracked in this repository.
