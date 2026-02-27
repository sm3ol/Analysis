# AV Encoder Subrepo (LiDAR)

This folder packages the AV LiDAR encoder path.

Included encoders:
- `pointpillars`
- `pointrcnn`
- `pv_rcnn`
- `centerpoint`

Core files:
- `taxonomy_metrics_av_lidar.py`
- `configs/av_taxonomy_manifest.json`
- `run_pointpillars.sh`, `run_pointrcnn.sh`, `run_pv_rcnn.sh`, `run_centerpoint.sh`
- `tools/generate_synthetic_lidar_data.py`
- `smoke_test.sh`

## Environment

```bash
bash setup_env.sh
```

## Usage

Each runner expects nuScenes data root:
```bash
bash run_pointpillars.sh /path/to/nuscenes
bash run_pointrcnn.sh /path/to/nuscenes
bash run_pv_rcnn.sh /path/to/nuscenes
bash run_centerpoint.sh /path/to/nuscenes
```

## Smoke Test (Synthetic Data)

```bash
bash smoke_test.sh
```

What it does:
- generates tiny synthetic LiDAR `.bin` files under `smoke_data/nuscenes_mock/samples/LIDAR_TOP`
- runs each encoder separately
- verifies `metrics_table.csv` is produced

## Data Requirements

- Full runs need nuScenes-style LiDAR files under `<data_root>/samples/LIDAR_TOP/*.bin`.
- nuScenes data is not tracked in this repository.
