#!/usr/bin/env python
"""Taxonomy-oriented family similarity metrics on MedMNIST-C-style data.

This script computes representation-shift metrics based on:
    Delta = rep(corrupted) - rep(clean)

Requested features:
- Delta normalization variants: raw + l2-normalized
- Metrics pooled across datasets + per-dataset
- Robust Delta aggregation: mean / median / trimmed_mean / per_image
- Matched-index consistency checks between clean and corrupted files
- Extra metrics: ARI/NMI and pairwise AUROC/AUPRC

Outputs (under run directory):
- config.json
- points_metadata.csv
- matched_index_checks.csv
- metrics_table.csv
- details.json

Data assumptions for this script:
- clean root contains MedMNIST 224 npz files (e.g., breastmnist_224.npz)
- corrupted root contains generated files per dataset/corruption (e.g., breastmnist/pixelate.npz)
- each corruption npz contains stacked severities in test_images/test_labels with 5 contiguous blocks
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
try:
    import torch

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False
from scipy.stats import trim_mean
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    f1_score,
    normalized_mutual_info_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import pairwise_distances


DEFAULT_FAMILY_MAP = {
    "digital": ["pixelate", "jpeg_compression"],
    "noise": ["gaussian_noise", "speckle_noise", "impulse_noise", "shot_noise"],
    "blur": ["defocus_blur", "motion_blur", "zoom_blur", "gaussian_blur"],
    "color": ["brightness_up", "brightness_down", "contrast_up", "contrast_down", "saturate"],
    "task-specific": [
        "stain_deposit",
        "bubble",
        "black_corner",
        "characters",
        "gamma_corr_up",
        "gamma_corr_down",
    ],
}


ENCODER_REGISTRY = {
    "clip_vit": ("ClipZeroShot", "vit"),
    "clip_resnet": ("ClipZeroShot", "resnet"),
    "biomedclip_vit": ("BioMedClipZeroShot", "vit"),
    "medclip_vit": ("MedclipZeroShot", "vit"),
    "medclip_resnet": ("MedclipZeroShot", "resnet"),
}


@dataclass
class PointMeta:
    point_id: str
    dataset: str
    corruption: str
    severity: int
    family: str
    sample_index: Optional[int]
    aggregation: str


@dataclass
class MatchedCheck:
    dataset: str
    split: str
    corruption: str
    severity: int
    selected_count: int
    labels_match: bool
    mismatch_count: int


class _MockVisionBackbone:
    def __init__(self, seed: int, out_dim: int = 64):
        self.seed = seed
        self.out_dim = out_dim

    def encode_image(self, x):
        if TORCH_AVAILABLE and torch is not None and isinstance(x, torch.Tensor):
            arr = x.detach().cpu().numpy()
        else:
            arr = np.asarray(x)
        if arr.ndim != 4:
            raise ValueError(f"Expected 4D image tensor, got {arr.shape}")
        pooled = arr.mean(axis=(2, 3))
        rng = np.random.default_rng(self.seed)
        proj = rng.standard_normal((pooled.shape[1], self.out_dim), dtype=np.float32)
        out = pooled.astype(np.float32) @ proj
        if TORCH_AVAILABLE and torch is not None:
            return torch.from_numpy(out)
        return out


class _MockEncoder:
    def __init__(self, name: str):
        self.name = name
        self.device = "cpu"
        self.preprocess = None
        self.is_mock = True
        self.model = _MockVisionBackbone(seed=sum(ord(c) for c in name) % 10007)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MedMNIST-C taxonomy metrics")
    p.add_argument(
        "--clean_root",
        type=str,
        default="../medmnistc-api/data/medmnist_224",
        help="Path to clean MedMNIST 224 npz files",
    )
    p.add_argument(
        "--corrupted_root",
        type=str,
        default="../medmnistc-api/data/medmnistc_generated",
        help="Path to generated MedMNIST-C corruption npz files",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="outputs/taxonomy_medmnistc",
        help="Output directory root",
    )
    p.add_argument(
        "--encoders",
        type=str,
        default="clip_vit,biomedclip_vit,clip_resnet,medclip_resnet",
        help="Comma-separated encoder keys",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference (default: cuda if available else cpu)",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="Comma list or 'all'",
    )
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test"],
    )
    p.add_argument(
        "--corruptions",
        type=str,
        default="all",
        help="Comma list or 'all'",
    )
    p.add_argument(
        "--severities",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated severity integers",
    )
    p.add_argument(
        "--max_samples_per_dataset",
        type=int,
        default=512,
        help="Max clean/corrupted matched samples per dataset",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Image encoder batch size",
    )
    p.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="Square image size fed into encoders (applies safety resize when needed)",
    )
    p.add_argument(
        "--delta_aggregation",
        type=str,
        default="mean",
        choices=["mean", "median", "trimmed_mean", "per_image"],
    )
    p.add_argument(
        "--trim_alpha",
        type=float,
        default=0.1,
        help="Trim proportion for trimmed_mean",
    )
    p.add_argument(
        "--per_image_samples",
        type=int,
        default=8,
        help="Per group sample count when --delta_aggregation=per_image",
    )
    p.add_argument(
        "--family_map_json",
        type=str,
        default=None,
        help="Optional JSON file mapping family -> [corruptions]",
    )
    p.add_argument(
        "--taxonomy_mode",
        type=str,
        default="fixed",
        choices=["fixed", "discover_then_validate"],
        help="Use fixed family map or discover taxonomy then validate on held-out severities",
    )
    p.add_argument(
        "--discovery_severities",
        type=str,
        default="1,2,3",
        help="Severities used to discover taxonomy (discover_then_validate mode)",
    )
    p.add_argument(
        "--validation_severities",
        type=str,
        default="4,5",
        help="Severities used for validation metrics (discover_then_validate mode)",
    )
    p.add_argument(
        "--evaluate_on",
        type=str,
        default="validation",
        choices=["validation", "all"],
        help="Evaluate metrics on validation severities only or on all severities",
    )
    p.add_argument(
        "--cluster_method",
        type=str,
        default="kmeans",
        choices=["kmeans", "agglomerative_cosine"],
        help="Clustering method for taxonomy discovery",
    )
    p.add_argument(
        "--n_discovery_families",
        type=int,
        default=0,
        help="If >0, force number of discovered families; else choose by silhouette",
    )
    p.add_argument(
        "--k_min",
        type=int,
        default=2,
        help="Minimum K for automatic family count selection",
    )
    p.add_argument(
        "--k_max",
        type=int,
        default=8,
        help="Maximum K for automatic family count selection",
    )
    p.add_argument(
        "--unknown_family_mode",
        type=str,
        default="error",
        choices=["error", "skip", "as_own_family"],
    )
    p.add_argument("--topk_dims", type=int, default=50)
    p.add_argument("--knn_ks", type=str, default="1,3,5")
    p.add_argument("--num_perm", type=int, default=1000)
    p.add_argument(
        "--max_pairs",
        type=int,
        default=500_000,
        help="Max pair count used for pair-based metrics/permutation",
    )
    p.add_argument(
        "--strict_index_check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if clean/corrupted labels do not match for selected indices",
    )
    p.add_argument(
        "--mock_encoder_preflight",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use deterministic mock encoders for smoke tests (no model downloads).",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _parse_csv_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _parse_int_list(value: str) -> List[int]:
    out: List[int] = []
    for part in _parse_csv_list(value):
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out


def _load_family_map(path: Optional[str]) -> Dict[str, List[str]]:
    if path is None:
        return DEFAULT_FAMILY_MAP
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("family_map_json must be a dict: family -> list[corruption]")
    out: Dict[str, List[str]] = {}
    for k, v in data.items():
        if not isinstance(v, list):
            raise ValueError(f"family '{k}' must map to list")
        out[str(k)] = [str(x) for x in v]
    return out


def _invert_family_map(family_map: Dict[str, List[str]]) -> Dict[str, str]:
    inv: Dict[str, str] = {}
    for fam, corrs in family_map.items():
        for c in corrs:
            if c in inv and inv[c] != fam:
                raise ValueError(f"Corruption '{c}' is assigned to multiple families")
            inv[c] = fam
    return inv


def _list_datasets(corrupted_root: Path) -> List[str]:
    datasets: List[str] = []
    for p in sorted(corrupted_root.iterdir()):
        if p.is_dir() and not p.name.startswith("."):
            datasets.append(p.name)
    return datasets


def _discover_corruptions(dataset_dir: Path) -> List[str]:
    return sorted([f.stem for f in dataset_dir.glob("*.npz") if f.is_file()])


def _resolve_clean_npz(clean_root: Path, dataset: str) -> Path:
    candidates = [
        clean_root / f"{dataset}_224.npz",
        clean_root / f"{dataset}.npz",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Clean MedMNIST npz not found for dataset '{dataset}'. Tried: "
        + ", ".join(str(x) for x in candidates)
    )


def _load_clean_split_npz(path: Path, split: str) -> Tuple[np.ndarray, np.ndarray]:
    npz = np.load(path, mmap_mode="r")
    return npz[f"{split}_images"], npz[f"{split}_labels"]


def _load_corruption_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    npz = np.load(path, mmap_mode="r")
    return npz["test_images"], npz["test_labels"]


def _ensure_hwc3(images: np.ndarray) -> np.ndarray:
    if images.ndim == 3:
        images = images[..., None]
    if images.ndim != 4:
        raise ValueError(f"Expected image array with 3 or 4 dims, got shape {images.shape}")
    if images.shape[-1] == 1:
        images = np.repeat(images, 3, axis=-1)
    elif images.shape[-1] != 3:
        raise ValueError(f"Unsupported channel count in shape {images.shape}")
    if images.dtype != np.uint8:
        images = np.clip(images, 0, 255).astype(np.uint8)
    return images


def _slice_severity_block(
    corr_imgs: np.ndarray,
    corr_labels: np.ndarray,
    n_clean: int,
    severity: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if severity < 1:
        raise ValueError(f"Severity must be >=1, got {severity}")
    start = (severity - 1) * n_clean
    end = severity * n_clean
    if end > len(corr_imgs) or end > len(corr_labels):
        raise ValueError(
            f"Corruption file has insufficient samples for severity {severity}: "
            f"need [{start}:{end}), got {len(corr_imgs)} images / {len(corr_labels)} labels"
        )
    return corr_imgs[start:end], corr_labels[start:end]


def _select_indices(n: int, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if max_n <= 0 or max_n >= n:
        return np.arange(n, dtype=np.int64)
    idx = rng.choice(n, size=max_n, replace=False)
    return np.sort(idx.astype(np.int64))


def _check_matched_indices(
    clean_labels: np.ndarray,
    corr_labels: np.ndarray,
    sel: np.ndarray,
    dataset: str,
    split: str,
    corruption: str,
    severity: int,
    strict: bool,
) -> MatchedCheck:
    a = clean_labels[sel].reshape(len(sel), -1)
    b = corr_labels[sel].reshape(len(sel), -1)
    labels_match = bool(np.array_equal(a, b))
    mismatch = int(np.sum(a != b))
    check = MatchedCheck(
        dataset=dataset,
        split=split,
        corruption=corruption,
        severity=severity,
        selected_count=int(len(sel)),
        labels_match=labels_match,
        mismatch_count=mismatch,
    )
    if strict and not labels_match:
        raise ValueError(
            f"Matched-index check failed for {dataset}/{split}/{corruption}/sev{severity}: "
            f"{mismatch} mismatched label entries"
        )
    return check


def _instantiate_encoder(name: str, device: str, mock_encoder_preflight: bool = False):
    if name not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder '{name}'. Available: {sorted(ENCODER_REGISTRY)}")
    if mock_encoder_preflight:
        return _MockEncoder(name=name)
    cls_name, vision_cls = ENCODER_REGISTRY[name]
    from models import BioMedClipZeroShot, ClipZeroShot, MedclipZeroShot

    class_map = {
        "ClipZeroShot": ClipZeroShot,
        "BioMedClipZeroShot": BioMedClipZeroShot,
        "MedclipZeroShot": MedclipZeroShot,
    }
    cls = class_map[cls_name]
    model = cls(vision_cls=vision_cls, device=device)
    return model


def _encode_images(
    model,
    images_u8: np.ndarray,
    batch_size: int,
    input_size: int,
) -> np.ndarray:
    if getattr(model, "is_mock", False):
        arr = images_u8.astype(np.float32) / 255.0
        arr = np.transpose(arr, (0, 3, 1, 2))
        out = model.model.encode_image(arr)
        if TORCH_AVAILABLE and torch is not None and isinstance(out, torch.Tensor):
            out = out.detach().cpu().numpy()
        out = np.asarray(out, dtype=np.float32)
        norm = np.linalg.norm(out, axis=-1, keepdims=True)
        return out / np.maximum(norm, 1e-12)

    if not TORCH_AVAILABLE or torch is None:
        raise RuntimeError(
            "torch is required for real encoder inference. Install dependencies or pass --mock_encoder_preflight."
        )

    feats: List[np.ndarray] = []
    n = images_u8.shape[0]
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        batch_u8 = images_u8[start:end]
        batch = torch.from_numpy(batch_u8).permute(0, 3, 1, 2).contiguous()
        batch = batch.float() / 255.0

        # Safety guard for mixed-resolution corruption files (e.g., pixelate can be 32x32).
        if batch.shape[-2] != input_size or batch.shape[-1] != input_size:
            batch = torch.nn.functional.interpolate(
                batch,
                size=(input_size, input_size),
                mode="bilinear",
                align_corners=False,
            )

        if model.preprocess is not None:
            batch = model.preprocess(batch)

        batch = batch.to(model.device)
        with torch.no_grad():
            try:
                out = model.model.encode_image(batch)
            except TypeError:
                out = model.model.encode_image(pixel_values=batch)

            if isinstance(out, dict):
                if "pooler_output" in out:
                    out = out["pooler_output"]
                elif "last_hidden_state" in out:
                    out = out["last_hidden_state"].mean(dim=1)
                else:
                    raise ValueError("Unsupported encoder output dict format")
            elif isinstance(out, (list, tuple)):
                out = out[0]

            out = out.float()
            out = out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            feats.append(out.cpu().numpy())

    return np.concatenate(feats, axis=0)


def _aggregate_delta(delta: np.ndarray, mode: str, trim_alpha: float) -> np.ndarray:
    if mode == "mean":
        return delta.mean(axis=0)
    if mode == "median":
        return np.median(delta, axis=0)
    if mode == "trimmed_mean":
        return trim_mean(delta, proportiontocut=trim_alpha, axis=0)
    raise ValueError(f"Unsupported aggregation mode '{mode}'")


def _row_l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return x / denom


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    if u == 0:
        return 0.0
    return len(a & b) / float(u)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    pooled = ((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2)
    if pooled <= 0:
        return float("nan")
    return (float(np.mean(b)) - float(np.mean(a))) / math.sqrt(pooled)


def _safe_silhouette(x: np.ndarray, labels: np.ndarray, metric: str) -> float:
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2:
        return float("nan")
    if np.min(counts) < 2:
        return float("nan")
    try:
        return float(silhouette_score(x, labels, metric=metric))
    except Exception:
        return float("nan")


def _family_to_int(labels: Sequence[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    uniq = sorted(set(labels))
    mapping = {k: i for i, k in enumerate(uniq)}
    arr = np.array([mapping[x] for x in labels], dtype=np.int32)
    return arr, mapping


def _fit_cluster_labels(
    x: np.ndarray,
    n_clusters: int,
    method: str,
    seed: int,
) -> np.ndarray:
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        return model.fit_predict(x)
    if method == "agglomerative_cosine":
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",
        )
        return model.fit_predict(x)
    raise ValueError(f"Unsupported cluster method: {method}")


def _discover_taxonomy_from_points(
    x_raw: np.ndarray,
    metas: Sequence[PointMeta],
    discovery_severities: Sequence[int],
    cluster_method: str,
    n_discovery_families: int,
    k_min: int,
    k_max: int,
    seed: int,
) -> Dict[str, object]:
    sev_set = set(int(s) for s in discovery_severities)
    idx_disc = np.array([i for i, m in enumerate(metas) if int(m.severity) in sev_set], dtype=np.int64)
    if len(idx_disc) == 0:
        raise ValueError("No points available for discovery severities")

    by_corr: Dict[str, List[int]] = {}
    for i in idx_disc.tolist():
        by_corr.setdefault(metas[i].corruption, []).append(i)
    corr_names = sorted(by_corr.keys())
    if len(corr_names) < 2:
        raise ValueError("Need at least 2 corruption types to discover taxonomy")

    prototypes = np.stack([x_raw[np.array(by_corr[c], dtype=np.int64)].mean(axis=0) for c in corr_names], axis=0)
    n_corr = len(corr_names)

    if n_discovery_families > 0:
        k = int(n_discovery_families)
        if k < 2 or k > n_corr:
            raise ValueError(
                f"n_discovery_families must be in [2, {n_corr}] for current discovery set; got {k}"
            )
        labels = _fit_cluster_labels(prototypes, n_clusters=k, method=cluster_method, seed=seed)
        sil = float("nan")
        if len(np.unique(labels)) > 1:
            try:
                sil = float(silhouette_score(prototypes, labels, metric="cosine"))
            except Exception:
                sil = float("nan")
        selected_k = k
        k_scores = [{"k": k, "silhouette_cosine": sil}]
    else:
        k_low = max(2, int(k_min))
        k_high = min(int(k_max), max(2, n_corr - 1))
        if k_high < k_low:
            k_low = 2
            k_high = max(2, n_corr - 1)
        if k_high < 2:
            raise ValueError("Not enough corruption prototypes to select K")

        best = None
        k_scores = []
        for k in range(k_low, k_high + 1):
            labels_k = _fit_cluster_labels(prototypes, n_clusters=k, method=cluster_method, seed=seed)
            if len(np.unique(labels_k)) < 2:
                sil = float("nan")
            else:
                try:
                    sil = float(silhouette_score(prototypes, labels_k, metric="cosine"))
                except Exception:
                    sil = float("nan")
            k_scores.append({"k": k, "silhouette_cosine": sil})
            score_cmp = -1e18 if not np.isfinite(sil) else sil
            best_cmp = -1e18 if (best is None or not np.isfinite(best["silhouette_cosine"])) else best["silhouette_cosine"]
            if best is None or score_cmp > best_cmp:
                best = {"k": k, "labels": labels_k, "silhouette_cosine": sil}
        selected_k = int(best["k"])
        labels = best["labels"]

    unique_labels = sorted(set(int(v) for v in labels.tolist()))
    old_to_new = {old: i for i, old in enumerate(unique_labels)}

    corruption_to_family: Dict[str, str] = {}
    family_to_corruptions: Dict[str, List[str]] = {}
    for c, lbl in zip(corr_names, labels.tolist()):
        fam = f"discovered_family_{old_to_new[int(lbl)]:02d}"
        corruption_to_family[c] = fam
        family_to_corruptions.setdefault(fam, []).append(c)

    for fam in family_to_corruptions:
        family_to_corruptions[fam] = sorted(family_to_corruptions[fam])

    return {
        "discovery_severities": sorted(sev_set),
        "n_corruptions_discovered": int(n_corr),
        "selected_k": int(selected_k),
        "cluster_method": cluster_method,
        "k_silhouette_scores": k_scores,
        "corruption_to_family": corruption_to_family,
        "family_to_corruptions": family_to_corruptions,
    }


def _compute_unsupervised_recoverability(
    x: np.ndarray,
    labels: Sequence[str],
    seed: int,
) -> Dict[str, float]:
    y, mapping = _family_to_int(labels)
    n_clusters = len(mapping)
    out = {
        "ari_kmeans": float("nan"),
        "nmi_kmeans": float("nan"),
        "ari_agglomerative_cosine": float("nan"),
        "nmi_agglomerative_cosine": float("nan"),
    }
    if n_clusters < 2 or x.shape[0] < n_clusters:
        return out

    try:
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        pred = km.fit_predict(x)
        out["ari_kmeans"] = float(adjusted_rand_score(y, pred))
        out["nmi_kmeans"] = float(normalized_mutual_info_score(y, pred))
    except Exception:
        pass

    try:
        agg = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",
        )
        pred = agg.fit_predict(x)
        out["ari_agglomerative_cosine"] = float(adjusted_rand_score(y, pred))
        out["nmi_agglomerative_cosine"] = float(normalized_mutual_info_score(y, pred))
    except Exception:
        pass

    return out


def _compute_distance_metrics(
    x: np.ndarray,
    labels: Sequence[str],
    point_ids: Sequence[str],
    metric_name: str,
    ks: Sequence[int],
    topk_dims: int,
    num_perm: int,
    max_pairs: int,
    seed: int,
) -> Dict[str, object]:
    metric = "euclidean" if metric_name == "l2" else "cosine"
    labels_arr = np.array(labels)
    n = x.shape[0]

    dist = pairwise_distances(x, metric=metric)
    tri_i, tri_j = np.triu_indices(n, 1)
    tri_d = dist[tri_i, tri_j]

    if len(tri_d) > max_pairs > 0:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(tri_d), size=max_pairs, replace=False))
        tri_i_s = tri_i[keep]
        tri_j_s = tri_j[keep]
        tri_d_s = tri_d[keep]
    else:
        tri_i_s, tri_j_s, tri_d_s = tri_i, tri_j, tri_d

    same = labels_arr[tri_i_s] == labels_arr[tri_j_s]
    within = tri_d_s[same]
    between = tri_d_s[~same]

    within_mean = float(np.mean(within)) if len(within) else float("nan")
    between_mean = float(np.mean(between)) if len(between) else float("nan")
    separation_margin = (
        float(between_mean - within_mean)
        if np.isfinite(within_mean) and np.isfinite(between_mean)
        else float("nan")
    )

    # Pairwise verification: same-family as positive; smaller distance -> larger score via -d.
    y_true = same.astype(np.int32)
    y_score = -tri_d_s
    if y_true.min() == y_true.max():
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = float(roc_auc_score(y_true, y_score))
        auprc = float(average_precision_score(y_true, y_score))

    # Permutation test on separation margin.
    perm_rng = np.random.default_rng(seed + 13)
    perm_margins: List[float] = []
    for _ in range(num_perm):
        shuffled = perm_rng.permutation(labels_arr)
        sm = shuffled[tri_i_s] == shuffled[tri_j_s]
        w = tri_d_s[sm]
        b = tri_d_s[~sm]
        if len(w) == 0 or len(b) == 0:
            continue
        perm_margins.append(float(np.mean(b) - np.mean(w)))
    if perm_margins and np.isfinite(separation_margin):
        count_ge = sum(m >= separation_margin for m in perm_margins)
        perm_p = float((count_ge + 1) / (len(perm_margins) + 1))
    else:
        perm_p = float("nan")

    # kNN purity.
    knn_metrics: Dict[str, float] = {}
    for k in ks:
        if k <= 0 or k >= n:
            knn_metrics[f"knn_purity_mean@{k}"] = float("nan")
            knn_metrics[f"knn_purity_median@{k}"] = float("nan")
            continue
        purities = []
        for i in range(n):
            idx = np.argsort(dist[i])
            idx = idx[idx != i][:k]
            purities.append(float(np.mean(labels_arr[idx] == labels_arr[i])))
        knn_metrics[f"knn_purity_mean@{k}"] = float(np.mean(purities))
        knn_metrics[f"knn_purity_median@{k}"] = float(np.median(purities))

    # Leave-one-out centroid accuracy and macro-F1.
    fam_to_idx: Dict[str, np.ndarray] = {}
    for fam in sorted(set(labels_arr.tolist())):
        fam_to_idx[fam] = np.where(labels_arr == fam)[0]

    pred_labels: List[str] = []
    true_labels: List[str] = []
    for i in range(n):
        x_i = x[i]
        best_fam = None
        best_dist = float("inf")
        for fam, idxs in fam_to_idx.items():
            use = idxs[idxs != i]
            if len(use) == 0:
                continue
            c = x[use].mean(axis=0)
            if metric_name == "l2":
                d = float(np.linalg.norm(x_i - c))
            else:
                denom = (np.linalg.norm(x_i) * np.linalg.norm(c))
                if denom <= 1e-12:
                    d = 1.0
                else:
                    d = float(1.0 - np.dot(x_i, c) / denom)
            if d < best_dist:
                best_dist = d
                best_fam = fam
        if best_fam is not None:
            pred_labels.append(best_fam)
            true_labels.append(labels_arr[i])

    if true_labels:
        loo_acc = float(np.mean(np.array(pred_labels) == np.array(true_labels)))
        loo_macro_f1 = float(f1_score(true_labels, pred_labels, average="macro"))
    else:
        loo_acc = float("nan")
        loo_macro_f1 = float("nan")

    # Top-K affected dimension overlap (Jaccard), within vs between.
    signatures: List[set] = []
    kdim = min(topk_dims, x.shape[1])
    for i in range(n):
        idx = np.argpartition(np.abs(x[i]), -kdim)[-kdim:]
        signatures.append(set(int(v) for v in idx.tolist()))

    jac_within: List[float] = []
    jac_between: List[float] = []
    for ii, jj in zip(tri_i_s, tri_j_s):
        jv = _jaccard(signatures[int(ii)], signatures[int(jj)])
        if labels_arr[int(ii)] == labels_arr[int(jj)]:
            jac_within.append(jv)
        else:
            jac_between.append(jv)
    jac_within_mean = float(np.mean(jac_within)) if jac_within else float("nan")
    jac_between_mean = float(np.mean(jac_between)) if jac_between else float("nan")
    jac_gap = (
        float(jac_within_mean - jac_between_mean)
        if np.isfinite(jac_within_mean) and np.isfinite(jac_between_mean)
        else float("nan")
    )

    # Misfit diagnostics.
    misfits = []
    misfit_flags = []
    for i in range(n):
        idx = np.argsort(dist[i])
        idx = idx[idx != i]
        if len(idx) == 0:
            continue
        nn = int(idx[0])
        bad = labels_arr[nn] != labels_arr[i]
        misfit_flags.append(bool(bad))
        if bad:
            misfits.append(
                {
                    "point_id": point_ids[i],
                    "family": str(labels_arr[i]),
                    "nearest_point_id": point_ids[nn],
                    "nearest_family": str(labels_arr[nn]),
                    "distance": float(dist[i, nn]),
                }
            )

    misfit_rate = float(np.mean(misfit_flags)) if misfit_flags else float("nan")
    misfit_per_family: Dict[str, float] = {}
    for fam in sorted(set(labels_arr.tolist())):
        fam_idx = np.where(labels_arr == fam)[0]
        fam_flags = []
        for i in fam_idx:
            idx = np.argsort(dist[i])
            idx = idx[idx != i]
            if len(idx) == 0:
                continue
            fam_flags.append(bool(labels_arr[int(idx[0])] != labels_arr[i]))
        misfit_per_family[fam] = float(np.mean(fam_flags)) if fam_flags else float("nan")

    metrics: Dict[str, object] = {
        "n_points": int(n),
        "pair_count_used": int(len(tri_d_s)),
        "pair_count_within": int(len(within)),
        "pair_count_between": int(len(between)),
        "within_mean_distance": within_mean,
        "between_mean_distance": between_mean,
        "separation_margin_between_minus_within": separation_margin,
        "effect_size_cohens_d": _cohens_d(within, between),
        "permutation_pvalue": perm_p,
        "silhouette_score": _safe_silhouette(x, labels_arr, metric=metric),
        "pairwise_verification_auroc": auroc,
        "pairwise_verification_auprc": auprc,
        "loo_centroid_accuracy": loo_acc,
        "loo_centroid_macro_f1": loo_macro_f1,
        "topk_overlap_jaccard_within_mean": jac_within_mean,
        "topk_overlap_jaccard_between_mean": jac_between_mean,
        "topk_overlap_jaccard_gap_within_minus_between": jac_gap,
        "misfit_rate": misfit_rate,
        "misfit_per_family": misfit_per_family,
        "misfit_examples": misfits[:25],
    }
    metrics.update(knn_metrics)
    return metrics


def _flatten_metric_rows(
    encoder: str,
    variant: str,
    scope: str,
    metric_key: str,
    metric_value: object,
    distance: str,
) -> Optional[Dict[str, object]]:
    if isinstance(metric_value, (dict, list)):
        return None
    if isinstance(metric_value, np.generic):
        metric_value = metric_value.item()
    return {
        "encoder": encoder,
        "delta_variant": variant,
        "scope": scope,
        "distance": distance,
        "metric": metric_key,
        "value": metric_value,
    }


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _build_points_for_encoder(
    model,
    clean_root: Path,
    corrupted_root: Path,
    datasets: Sequence[str],
    split: str,
    corruptions_by_dataset: Dict[str, Sequence[str]],
    severities: Sequence[int],
    family_of: Dict[str, str],
    aggregation: str,
    trim_alpha: float,
    per_image_samples: int,
    max_samples_per_dataset: int,
    strict_index_check: bool,
    unknown_family_mode: str,
    batch_size: int,
    input_size: int,
    seed: int,
) -> Tuple[np.ndarray, List[PointMeta], List[MatchedCheck], Dict[str, object]]:
    rng = np.random.default_rng(seed)

    vectors: List[np.ndarray] = []
    metas: List[PointMeta] = []
    checks: List[MatchedCheck] = []

    summary = {
        "datasets_processed": [],
        "groups_processed": 0,
        "groups_skipped": 0,
    }

    for dataset in datasets:
        try:
            clean_path = _resolve_clean_npz(clean_root, dataset)
        except FileNotFoundError:
            summary["groups_skipped"] += 1
            continue

        clean_imgs, clean_labels = _load_clean_split_npz(clean_path, split=split)
        clean_imgs = _ensure_hwc3(clean_imgs)
        sel = _select_indices(len(clean_imgs), max_samples_per_dataset, rng)
        clean_sel = clean_imgs[sel]
        clean_feat = _encode_images(
            model,
            clean_sel,
            batch_size=batch_size,
            input_size=input_size,
        )

        dataset_done = False
        dataset_dir = corrupted_root / dataset
        dataset_corrs = list(corruptions_by_dataset.get(dataset, []))
        for corr in dataset_corrs:
            fam = family_of.get(corr)
            if fam is None:
                if unknown_family_mode == "skip":
                    summary["groups_skipped"] += len(severities)
                    continue
                if unknown_family_mode == "as_own_family":
                    fam = corr
                else:
                    raise ValueError(f"No family mapping for corruption '{corr}'")

            corr_path = dataset_dir / f"{corr}.npz"
            if not corr_path.exists():
                summary["groups_skipped"] += len(severities)
                continue

            corr_imgs_all, corr_labels_all = _load_corruption_npz(corr_path)
            corr_imgs_all = _ensure_hwc3(corr_imgs_all)

            for sev in severities:
                corr_imgs, corr_labels = _slice_severity_block(
                    corr_imgs=corr_imgs_all,
                    corr_labels=corr_labels_all,
                    n_clean=len(clean_imgs),
                    severity=sev,
                )

                check = _check_matched_indices(
                    clean_labels=clean_labels,
                    corr_labels=corr_labels,
                    sel=sel,
                    dataset=dataset,
                    split=split,
                    corruption=corr,
                    severity=sev,
                    strict=strict_index_check,
                )
                checks.append(check)

                corr_sel = corr_imgs[sel]
                corr_feat = _encode_images(
                    model,
                    corr_sel,
                    batch_size=batch_size,
                    input_size=input_size,
                )

                delta = corr_feat - clean_feat

                if aggregation == "per_image":
                    keep = np.arange(delta.shape[0], dtype=np.int64)
                    if per_image_samples > 0 and per_image_samples < len(keep):
                        keep = np.sort(rng.choice(keep, size=per_image_samples, replace=False))
                    for ii in keep.tolist():
                        vectors.append(delta[ii].astype(np.float32))
                        metas.append(
                            PointMeta(
                                point_id=f"{dataset}|{corr}|s{sev}|i{int(sel[ii])}",
                                dataset=dataset,
                                corruption=corr,
                                severity=int(sev),
                                family=fam,
                                sample_index=int(sel[ii]),
                                aggregation=aggregation,
                            )
                        )
                else:
                    v = _aggregate_delta(delta, mode=aggregation, trim_alpha=trim_alpha)
                    vectors.append(v.astype(np.float32))
                    metas.append(
                        PointMeta(
                            point_id=f"{dataset}|{corr}|s{sev}",
                            dataset=dataset,
                            corruption=corr,
                            severity=int(sev),
                            family=fam,
                            sample_index=None,
                            aggregation=aggregation,
                        )
                    )

                summary["groups_processed"] += 1
                dataset_done = True

        if dataset_done:
            summary["datasets_processed"].append(dataset)

    if not vectors:
        raise ValueError("No valid Delta points were built. Check datasets/corruptions/severities.")

    x = np.stack(vectors, axis=0)
    return x, metas, checks, summary


def main() -> None:
    args = parse_args()

    clean_root = Path(args.clean_root)
    corrupted_root = Path(args.corrupted_root)
    if not clean_root.exists():
        raise FileNotFoundError(f"Clean root not found: {clean_root}")
    if not corrupted_root.exists():
        raise FileNotFoundError(f"Corrupted root not found: {corrupted_root}")

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.taxonomy_mode == "fixed":
        family_map = _load_family_map(args.family_map_json)
        family_of = _invert_family_map(family_map)
    else:
        family_map = {}
        family_of = {}

    all_datasets = _list_datasets(corrupted_root)
    if args.datasets == "all":
        datasets = all_datasets
    else:
        req = _parse_csv_list(args.datasets)
        missing = [m for m in req if m not in all_datasets]
        if missing:
            raise ValueError(f"Requested datasets not found under corrupted_root: {missing}")
        datasets = req

    if not datasets:
        raise ValueError("No datasets selected")

    requested_corruptions = _parse_csv_list(args.corruptions) if args.corruptions != "all" else []
    corruptions_by_dataset: Dict[str, List[str]] = {}
    for ds in datasets:
        available = _discover_corruptions(corrupted_root / ds)
        if args.corruptions == "all":
            selected = available
        else:
            selected = [c for c in requested_corruptions if c in set(available)]
        if not selected:
            print(f"[WARN] No selected corruptions found for dataset: {ds}")
        corruptions_by_dataset[ds] = selected

    severities = _parse_int_list(args.severities)
    if not severities:
        raise ValueError("No severities parsed from --severities")
    severity_set = set(severities)

    discovery_severities = _parse_int_list(args.discovery_severities)
    validation_severities = _parse_int_list(args.validation_severities)
    if args.taxonomy_mode == "discover_then_validate":
        if not discovery_severities:
            raise ValueError("discovery_severities must be non-empty in discover_then_validate mode")
        if not set(discovery_severities).issubset(severity_set):
            raise ValueError("discovery_severities must be subset of --severities")
        if validation_severities and not set(validation_severities).issubset(severity_set):
            raise ValueError("validation_severities must be subset of --severities")

    encoders = _parse_csv_list(args.encoders)
    if not encoders:
        raise ValueError("No encoders parsed from --encoders")

    ks = _parse_int_list(args.knn_ks)
    if not ks:
        raise ValueError("No valid k values parsed from --knn_ks")

    device = args.device or ("cuda" if (TORCH_AVAILABLE and torch is not None and torch.cuda.is_available()) else "cpu")

    config = {
        "clean_root": str(clean_root.resolve()),
        "corrupted_root": str(corrupted_root.resolve()),
        "output_dir": str(run_dir.resolve()),
        "encoders": encoders,
        "device": device,
        "datasets": datasets,
        "split": args.split,
        "corruptions_by_dataset": corruptions_by_dataset,
        "severities": severities,
        "max_samples_per_dataset": args.max_samples_per_dataset,
        "batch_size": args.batch_size,
        "input_size": args.input_size,
        "delta_aggregation": args.delta_aggregation,
        "trim_alpha": args.trim_alpha,
        "per_image_samples": args.per_image_samples,
        "taxonomy_mode": args.taxonomy_mode,
        "family_map": family_map if args.taxonomy_mode == "fixed" else None,
        "discovery_severities": discovery_severities,
        "validation_severities": validation_severities,
        "evaluate_on": args.evaluate_on,
        "cluster_method": args.cluster_method,
        "n_discovery_families": args.n_discovery_families,
        "k_min": args.k_min,
        "k_max": args.k_max,
        "unknown_family_mode": args.unknown_family_mode,
        "topk_dims": args.topk_dims,
        "knn_ks": ks,
        "num_perm": args.num_perm,
        "max_pairs": args.max_pairs,
        "strict_index_check": args.strict_index_check,
        "seed": args.seed,
    }

    details: Dict[str, object] = {
        "config": config,
        "encoders": {},
    }
    metric_rows: List[Dict[str, object]] = []
    points_rows_all: List[Dict[str, object]] = []
    checks_rows_all: List[Dict[str, object]] = []

    for enc in encoders:
        print(f"[INFO] Building points for encoder: {enc}")
        model = _instantiate_encoder(enc, device=device, mock_encoder_preflight=args.mock_encoder_preflight)

        mode_family_of = family_of if args.taxonomy_mode == "fixed" else {}
        mode_unknown = args.unknown_family_mode if args.taxonomy_mode == "fixed" else "as_own_family"

        x_raw, metas, checks, build_summary = _build_points_for_encoder(
            model=model,
            clean_root=clean_root,
            corrupted_root=corrupted_root,
            datasets=datasets,
            split=args.split,
            corruptions_by_dataset=corruptions_by_dataset,
            severities=severities,
            family_of=mode_family_of,
            aggregation=args.delta_aggregation,
            trim_alpha=args.trim_alpha,
            per_image_samples=args.per_image_samples,
            max_samples_per_dataset=args.max_samples_per_dataset,
            strict_index_check=args.strict_index_check,
            unknown_family_mode=mode_unknown,
            batch_size=args.batch_size,
            input_size=args.input_size,
            seed=args.seed,
        )

        discovery_info = None
        if args.taxonomy_mode == "discover_then_validate":
            discovery_info = _discover_taxonomy_from_points(
                x_raw=x_raw,
                metas=metas,
                discovery_severities=discovery_severities,
                cluster_method=args.cluster_method,
                n_discovery_families=args.n_discovery_families,
                k_min=args.k_min,
                k_max=args.k_max,
                seed=args.seed,
            )
            corr_to_family = discovery_info["corruption_to_family"]
            for m in metas:
                m.family = corr_to_family.get(m.corruption, m.corruption)

        sev_per_point = np.array([int(m.severity) for m in metas], dtype=np.int64)
        if args.taxonomy_mode == "discover_then_validate" and args.evaluate_on == "validation":
            eval_sev_set = set(int(s) for s in validation_severities)
            if not eval_sev_set:
                eval_sev_set = set(int(s) for s in severities) - set(int(s) for s in discovery_severities)
            if not eval_sev_set:
                raise ValueError("No evaluation severities available after discovery split")
            idx_eval = np.where(np.isin(sev_per_point, np.array(sorted(eval_sev_set), dtype=np.int64)))[0]
        else:
            idx_eval = np.arange(len(metas), dtype=np.int64)
            eval_sev_set = set(int(s) for s in severities)

        if len(idx_eval) < 3:
            raise ValueError(f"Too few evaluation points for encoder '{enc}': {len(idx_eval)}")

        eval_mask = np.zeros(len(metas), dtype=bool)
        eval_mask[idx_eval] = True
        eval_metas = [metas[i] for i in idx_eval.tolist()]

        x_variants = {
            "raw": x_raw,
            "l2_normalized": _row_l2_normalize(x_raw),
        }

        labels = [m.family for m in eval_metas]
        point_ids = [m.point_id for m in eval_metas]
        datasets_per_point = [m.dataset for m in eval_metas]

        for i, m in enumerate(metas):
            row = asdict(m)
            row["encoder"] = enc
            row["evaluation_included"] = bool(eval_mask[i])
            points_rows_all.append(row)

        for c in checks:
            row = asdict(c)
            row["encoder"] = enc
            checks_rows_all.append(row)

        enc_details = {
            "build_summary": build_summary,
            "n_points_total": int(len(metas)),
            "n_points_evaluated": int(len(eval_metas)),
            "evaluation_severities": sorted(int(s) for s in eval_sev_set),
            "taxonomy_mode": args.taxonomy_mode,
            "taxonomy_discovery": discovery_info,
            "variants": {},
        }

        scope_to_indices: Dict[str, np.ndarray] = {
            "pooled": np.arange(len(eval_metas), dtype=np.int64)
        }
        for ds in sorted(set(datasets_per_point)):
            idx = np.where(np.array(datasets_per_point) == ds)[0]
            scope_to_indices[f"dataset:{ds}"] = idx

        for var_name, x in x_variants.items():
            var_out: Dict[str, object] = {}
            x_eval = x[idx_eval]
            for scope_name, idx in scope_to_indices.items():
                if len(idx) < 3:
                    continue

                x_s = x_eval[idx]
                labels_s = [labels[i] for i in idx.tolist()]
                ids_s = [point_ids[i] for i in idx.tolist()]

                unsup = _compute_unsupervised_recoverability(
                    x=x_s,
                    labels=labels_s,
                    seed=args.seed,
                )

                dist_out: Dict[str, object] = {}
                for dist_name in ("l2", "cosine"):
                    metrics = _compute_distance_metrics(
                        x=x_s,
                        labels=labels_s,
                        point_ids=ids_s,
                        metric_name=dist_name,
                        ks=ks,
                        topk_dims=args.topk_dims,
                        num_perm=args.num_perm,
                        max_pairs=args.max_pairs,
                        seed=args.seed,
                    )
                    dist_out[dist_name] = metrics
                    for mk, mv in metrics.items():
                        row = _flatten_metric_rows(
                            encoder=enc,
                            variant=var_name,
                            scope=scope_name,
                            metric_key=mk,
                            metric_value=mv,
                            distance=dist_name,
                        )
                        if row is not None:
                            metric_rows.append(row)

                scope_out = {
                    "unsupervised_recoverability": unsup,
                    "distance_metrics": dist_out,
                }
                var_out[scope_name] = scope_out

                for mk, mv in unsup.items():
                    row = _flatten_metric_rows(
                        encoder=enc,
                        variant=var_name,
                        scope=scope_name,
                        metric_key=mk,
                        metric_value=mv,
                        distance="n/a",
                    )
                    if row is not None:
                        metric_rows.append(row)

            enc_details["variants"][var_name] = var_out

        details["encoders"][enc] = enc_details

        del model
        if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    with (run_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    _write_csv(run_dir / "points_metadata.csv", points_rows_all)
    _write_csv(run_dir / "matched_index_checks.csv", checks_rows_all)
    _write_csv(run_dir / "metrics_table.csv", metric_rows)

    with (run_dir / "details.json").open("w") as f:
        json.dump(details, f, indent=2, default=_json_default)

    print(f"[DONE] Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
