#!/usr/bin/env python
"""AV LiDAR taxonomy metrics runner.

Builds feature-shift points from nuScenes LiDAR frame windows under synthetic
corruptions and exports taxonomy metrics in a metrics_table.csv compatible
with run_full_taxonomy_rerun_and_report_av.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    SKLEARN_AVAILABLE = True
except Exception:
    KMeans = None
    adjusted_rand_score = None
    normalized_mutual_info_score = None
    SKLEARN_AVAILABLE = False


DEFAULT_METRICS = [
    "separation_margin_between_minus_within",
    "pairwise_verification_auroc",
    "pairwise_verification_auprc",
    "silhouette_score",
    "ari_kmeans",
    "nmi_kmeans",
]

ALL_LIDAR_CORRUPTIONS = [
    "density_decrease",
    "cutout",
    "fov_lost",
    "lidar_crosstalk",
    "motion_compensation",
    "spatial_misalignment",
    "temporal_misalignment",
    "rotation",
    "scale",
    "shear",
    "moving_object",
    "fog",
    "rain",
    "snow",
    "strong_sunlight",
    "local_density_decrease",
    "local_cutout",
    "local_gaussian_noise",
    "local_uniform_noise",
    "local_impulse_noise",
]

FAMILY_OF = {
    "density_decrease": "dropout",
    "cutout": "dropout",
    "fov_lost": "dropout",
    "local_density_decrease": "local_noise",
    "local_cutout": "local_noise",
    "local_gaussian_noise": "local_noise",
    "local_uniform_noise": "local_noise",
    "local_impulse_noise": "local_noise",
    "lidar_crosstalk": "sensor_noise",
    "motion_compensation": "misalignment",
    "spatial_misalignment": "misalignment",
    "temporal_misalignment": "misalignment",
    "rotation": "geometry",
    "scale": "geometry",
    "shear": "geometry",
    "moving_object": "dynamic",
    "fog": "weather",
    "rain": "weather",
    "snow": "weather",
    "strong_sunlight": "illumination",
}


def _parse_csv_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _parse_int_list(value: str) -> List[int]:
    return [int(x) for x in _parse_csv_list(value)]


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _safe_silhouette(x: np.ndarray, labels: np.ndarray, metric_name: str) -> float:
    if x.shape[0] < 3:
        return float("nan")
    if len(set(labels.tolist())) < 2:
        return float("nan")
    try:
        d = _pairwise_distances(x, metric_name)
        n = len(labels)
        uniq = sorted(set(labels.tolist()))
        if len(uniq) < 2 or n < 3:
            return float("nan")
        s_vals = []
        for i in range(n):
            same = labels == labels[i]
            same[i] = False
            a = float(np.mean(d[i, same])) if np.any(same) else 0.0
            b = float("inf")
            for u in uniq:
                if u == labels[i]:
                    continue
                m = labels == u
                if np.any(m):
                    b = min(b, float(np.mean(d[i, m])))
            if not np.isfinite(b):
                continue
            den = max(a, b, 1e-12)
            s_vals.append((b - a) / den)
        return float(np.mean(s_vals)) if s_vals else float("nan")
    except Exception:
        return float("nan")


def _clustering_metrics(x: np.ndarray, labels: np.ndarray, seed: int) -> Dict[str, float]:
    if not SKLEARN_AVAILABLE:
        return {"ari_kmeans": float("nan"), "nmi_kmeans": float("nan")}
    uniq = sorted(set(labels.tolist()))
    if len(uniq) < 2 or x.shape[0] < 3:
        return {"ari_kmeans": float("nan"), "nmi_kmeans": float("nan")}
    k = min(len(uniq), max(2, min(8, x.shape[0] - 1)))
    try:
        pred = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(x)  # type: ignore[misc]
        return {
            "ari_kmeans": float(adjusted_rand_score(labels, pred)),
            "nmi_kmeans": float(normalized_mutual_info_score(labels, pred)),
        }
    except Exception:
        return {"ari_kmeans": float("nan"), "nmi_kmeans": float("nan")}


def _pairwise_distances(x: np.ndarray, metric_name: str) -> np.ndarray:
    if metric_name == "l2":
        g = np.sum(x * x, axis=1, keepdims=True)
        d2 = np.maximum(g + g.T - 2.0 * (x @ x.T), 0.0)
        return np.sqrt(d2, dtype=np.float32)
    xn = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return (1.0 - np.clip(xn @ xn.T, -1.0, 1.0)).astype(np.float32)


def _binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    scores = np.concatenate([pos, neg])
    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    idx_pos = np.where(y_true == 1)[0]
    r_pos = ranks[idx_pos]
    u = np.sum(r_pos) - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def _binary_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    p = int(np.sum(y_true == 1))
    if p == 0:
        return float("nan")
    order = np.argsort(-y_score)
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    prec = tp / np.maximum(tp + fp, 1)
    return float(np.sum(prec[y == 1]) / p)


def _evaluate_space(
    x: np.ndarray,
    labels: Sequence[str],
    metric_name: str,
    max_pairs: int,
    seed: int,
) -> Dict[str, float]:
    if x.shape[0] < 3:
        return {m: float("nan") for m in DEFAULT_METRICS}
    labels_arr = np.array(labels)
    dist = _pairwise_distances(x, metric_name)
    tri_i, tri_j = np.triu_indices(x.shape[0], 1)
    tri_d = dist[tri_i, tri_j]
    if max_pairs > 0 and len(tri_d) > max_pairs:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(tri_d), size=max_pairs, replace=False))
        tri_i, tri_j, tri_d = tri_i[keep], tri_j[keep], tri_d[keep]

    same = labels_arr[tri_i] == labels_arr[tri_j]
    within = tri_d[same]
    between = tri_d[~same]
    within_mean = float(np.mean(within)) if len(within) else float("nan")
    between_mean = float(np.mean(between)) if len(between) else float("nan")
    separation = (
        float(between_mean - within_mean)
        if np.isfinite(within_mean) and np.isfinite(between_mean)
        else float("nan")
    )

    y_true = same.astype(np.int32)
    y_score = -tri_d
    if len(y_true) == 0 or y_true.min() == y_true.max():
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = _binary_auroc(y_true, y_score)
        auprc = _binary_auprc(y_true, y_score)

    clus = _clustering_metrics(x, labels_arr, seed=seed)
    return {
        "separation_margin_between_minus_within": separation,
        "pairwise_verification_auroc": auroc,
        "pairwise_verification_auprc": auprc,
        "silhouette_score": _safe_silhouette(x, labels_arr, metric_name),
        "ari_kmeans": clus["ari_kmeans"],
        "nmi_kmeans": clus["nmi_kmeans"],
    }


def _encoder_seed(name: str) -> int:
    h = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    return int(h, 16)


def _project_features(base: np.ndarray, encoder: str, out_dim: int = 128) -> np.ndarray:
    rng = np.random.default_rng(_encoder_seed(encoder))
    w = rng.standard_normal((base.shape[1], out_dim), dtype=np.float32)
    b = rng.standard_normal((out_dim,), dtype=np.float32) * 0.05
    x = np.tanh(base @ w + b)
    return x.astype(np.float32)


def _build_representation(base: np.ndarray, encoder: str, rep_mode: str) -> np.ndarray:
    if rep_mode == "pooled":
        return _project_features(base, encoder=encoder, out_dim=128)
    if rep_mode == "tokens":
        # Approximate token-level structure by chunk-wise projections.
        n_dim = base.shape[1]
        n_chunks = 8
        step = max(1, (n_dim + n_chunks - 1) // n_chunks)
        parts: List[np.ndarray] = []
        for ci in range(n_chunks):
            s = ci * step
            e = min(n_dim, (ci + 1) * step)
            if s >= e:
                break
            chunk = base[:, s:e]
            parts.append(_project_features(chunk, encoder=f"{encoder}::tok{ci}", out_dim=32))
        if not parts:
            return _project_features(base, encoder=f"{encoder}::tok_fallback", out_dim=128)
        return np.concatenate(parts, axis=1).astype(np.float32)
    raise ValueError(f"Unsupported rep_mode: {rep_mode}")


def _lidar_base_feature(points: np.ndarray) -> np.ndarray:
    xyz = points[:, :3].astype(np.float32)
    means = xyz.mean(axis=0)
    stds = xyz.std(axis=0)
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    r = np.linalg.norm(xyz[:, :2], axis=1)
    z = xyz[:, 2]
    az = np.arctan2(xyz[:, 1], xyz[:, 0])
    r_hist, _ = np.histogram(r, bins=16, range=(0.0, max(1.0, float(np.percentile(r, 99)))), density=True)
    z_hist, _ = np.histogram(z, bins=16, range=(float(np.percentile(z, 1)), float(np.percentile(z, 99))), density=True)
    a_hist, _ = np.histogram(az, bins=16, range=(-math.pi, math.pi), density=True)
    extras = [
        float(r.mean()),
        float(r.std()),
        float(np.percentile(r, 90)),
        float(np.percentile(z, 10)),
        float(np.percentile(z, 90)),
    ]
    if points.shape[1] > 3:
        inten = points[:, 3].astype(np.float32)
        extras.extend([float(inten.mean()), float(inten.std()), float(np.percentile(inten, 90))])
    feat = np.concatenate(
        [
            means,
            stds,
            mins,
            maxs,
            np.array(extras, dtype=np.float32),
            r_hist.astype(np.float32),
            z_hist.astype(np.float32),
            a_hist.astype(np.float32),
        ],
        axis=0,
    )
    return feat.astype(np.float32)


def _rot_z(points: np.ndarray, angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    out = points.copy()
    x = out[:, 0].copy()
    y = out[:, 1].copy()
    out[:, 0] = c * x - s * y
    out[:, 1] = s * x + c * y
    return out


def _apply_lidar_corruption(points: np.ndarray, corruption: str, severity: int, rng: np.random.Generator) -> np.ndarray:
    s = max(1, min(5, int(severity)))
    p = points.copy()
    if len(p) < 32:
        return p

    if corruption == "density_decrease":
        keep_prob = 1.0 - (0.08 + 0.08 * s)
        keep = rng.random(len(p)) < max(0.1, keep_prob)
        p = p[keep]
    elif corruption == "cutout":
        center = p[rng.integers(0, len(p)), :3]
        rad = 0.8 + 0.35 * s
        d = np.linalg.norm(p[:, :3] - center[None, :], axis=1)
        p = p[d > rad]
    elif corruption == "fov_lost":
        az = np.arctan2(p[:, 1], p[:, 0])
        w = math.radians(12 + 10 * s)
        c = rng.uniform(-math.pi, math.pi)
        d = np.angle(np.exp(1j * (az - c)))
        p = p[np.abs(d) > w / 2]
    elif corruption == "lidar_crosstalk":
        n_new = max(32, int(len(p) * (0.01 + 0.02 * s)))
        mins = p[:, :3].min(axis=0)
        maxs = p[:, :3].max(axis=0)
        noise_xyz = rng.uniform(mins, maxs, size=(n_new, 3))
        extra_dim = p.shape[1] - 3
        if extra_dim > 0:
            noise_extra = rng.normal(0.0, 1.0, size=(n_new, extra_dim)).astype(np.float32)
            q = np.concatenate([noise_xyz, noise_extra], axis=1)
        else:
            q = noise_xyz
        p = np.concatenate([p, q.astype(np.float32)], axis=0)
    elif corruption == "motion_compensation":
        az = np.arctan2(p[:, 1], p[:, 0])
        p[:, 0] += (0.01 * s) * np.sin(2.0 * az)
        p[:, 1] += (0.01 * s) * np.cos(2.0 * az)
    elif corruption == "spatial_misalignment":
        t = np.array([0.05 * s, -0.03 * s, 0.02 * s], dtype=np.float32)
        p[:, :3] += t[None, :]
    elif corruption == "temporal_misalignment":
        p = _rot_z(p, angle_rad=math.radians(1.5 * s))
        p[:, :3] += rng.normal(0.0, 0.01 * s, size=p[:, :3].shape).astype(np.float32)
    elif corruption == "rotation":
        p = _rot_z(p, angle_rad=math.radians(2.0 + 2.5 * s))
    elif corruption == "scale":
        k = 1.0 + 0.02 * s
        p[:, :3] *= k
    elif corruption == "shear":
        sh = 0.01 * s
        p[:, 0] = p[:, 0] + sh * p[:, 1]
    elif corruption == "moving_object":
        center = p[rng.integers(0, len(p)), :3]
        d = np.linalg.norm(p[:, :3] - center[None, :], axis=1)
        m = d < (0.7 + 0.25 * s)
        p[m, 0] += 0.03 * s
    elif corruption == "fog":
        r = np.linalg.norm(p[:, :2], axis=1)
        drop = rng.random(len(p)) < np.clip((r / (r.max() + 1e-6)) * (0.03 * s), 0, 0.7)
        p = p[~drop]
        if p.shape[1] > 3:
            p[:, 3] *= (1.0 - 0.08 * s)
    elif corruption == "rain":
        drop = rng.random(len(p)) < (0.02 + 0.03 * s)
        p = p[~drop]
        p[:, 2] += rng.normal(0.0, 0.005 * s, size=len(p)).astype(np.float32)
    elif corruption == "snow":
        p[:, 2] += rng.normal(0.0, 0.01 * s, size=len(p)).astype(np.float32)
        if p.shape[1] > 3:
            p[:, 3] *= (1.0 - 0.05 * s)
    elif corruption == "strong_sunlight":
        if p.shape[1] > 3:
            p[:, 3] = np.clip(p[:, 3] + rng.normal(0.0, 0.2 * s, size=len(p)), 0.0, None)
    elif corruption == "local_density_decrease":
        center = p[rng.integers(0, len(p)), :3]
        d = np.linalg.norm(p[:, :3] - center[None, :], axis=1)
        m = d < (1.0 + 0.3 * s)
        keep = np.ones(len(p), dtype=bool)
        drop_local = rng.random(m.sum()) < (0.1 + 0.1 * s)
        keep[np.where(m)[0][drop_local]] = False
        p = p[keep]
    elif corruption == "local_cutout":
        center = p[rng.integers(0, len(p)), :3]
        d = np.linalg.norm(p[:, :3] - center[None, :], axis=1)
        p = p[d > (0.8 + 0.25 * s)]
    elif corruption == "local_gaussian_noise":
        center = p[rng.integers(0, len(p)), :3]
        d = np.linalg.norm(p[:, :3] - center[None, :], axis=1)
        m = d < (1.1 + 0.25 * s)
        p[m, :3] += rng.normal(0.0, 0.015 * s, size=(m.sum(), 3)).astype(np.float32)
    elif corruption == "local_uniform_noise":
        center = p[rng.integers(0, len(p)), :3]
        d = np.linalg.norm(p[:, :3] - center[None, :], axis=1)
        m = d < (1.1 + 0.25 * s)
        p[m, :3] += rng.uniform(-0.02 * s, 0.02 * s, size=(m.sum(), 3)).astype(np.float32)
    elif corruption == "local_impulse_noise":
        center = p[rng.integers(0, len(p)), :3]
        d = np.linalg.norm(p[:, :3] - center[None, :], axis=1)
        idx = np.where(d < (1.0 + 0.2 * s))[0]
        if len(idx) > 0:
            k = max(1, int(0.2 * len(idx)))
            sel = rng.choice(idx, size=k, replace=False)
            mins = p[:, :3].min(axis=0)
            maxs = p[:, :3].max(axis=0)
            p[sel, :3] = rng.uniform(mins, maxs, size=(k, 3)).astype(np.float32)
    else:
        return points

    if len(p) < 32:
        return points
    return p


def _scope_subset(
    corr_name: str,
    by_corr: Dict[str, np.ndarray],
    fam_of: Dict[str, str],
    rng: np.random.Generator,
    cap_bg_per_family: int,
) -> Tuple[np.ndarray, List[str]]:
    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    fam_ref = fam_of[corr_name]
    x_ref = by_corr[corr_name]
    if len(x_ref) == 0:
        return np.zeros((0, 1), dtype=np.float32), []
    x_parts.append(x_ref)
    y_parts.append(np.array([fam_ref] * len(x_ref)))

    families = sorted(set(fam_of.values()))
    for fam in families:
        if fam == fam_ref:
            continue
        pool = [by_corr[c] for c in by_corr if fam_of[c] == fam and len(by_corr[c]) > 0]
        if not pool:
            continue
        xp = np.concatenate(pool, axis=0)
        if len(xp) > cap_bg_per_family:
            idx = rng.choice(len(xp), size=cap_bg_per_family, replace=False)
            xp = xp[idx]
        x_parts.append(xp)
        y_parts.append(np.array([fam] * len(xp)))

    x = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0).tolist()
    return x, y


def _vec_distance(a: np.ndarray, b: np.ndarray, metric_name: str) -> float:
    if metric_name == "cosine":
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return float(1.0 - (np.dot(a, b) / denom))
    return float(np.linalg.norm(a - b))


def _pairwise_type_matrix(type_list: Sequence[str], proto: Dict[str, np.ndarray], metric_name: str) -> np.ndarray:
    n = len(type_list)
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = _vec_distance(proto[type_list[i]], proto[type_list[j]], metric_name)
            mat[i, j] = d
            mat[j, i] = d
    return mat


def _within_between(type_list: Sequence[str], type_to_family: Dict[str, str], mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    within = []
    between = []
    for i in range(len(type_list)):
        for j in range(i + 1, len(type_list)):
            if type_to_family[type_list[i]] == type_to_family[type_list[j]]:
                within.append(float(mat[i, j]))
            else:
                between.append(float(mat[i, j]))
    return np.array(within, dtype=np.float32), np.array(between, dtype=np.float32)


def _effect_and_pvalue(
    within: np.ndarray,
    between: np.ndarray,
    type_list: Sequence[str],
    type_to_family: Dict[str, str],
    mat: np.ndarray,
    num_perm: int,
    seed: int,
) -> Tuple[float, float, float, float, float]:
    within_mean = float(np.mean(within)) if within.size else float("nan")
    between_mean = float(np.mean(between)) if between.size else float("nan")
    diff = between_mean - within_mean
    if within.size and between.size:
        pooled_std = float(np.sqrt(0.5 * (np.var(within) + np.var(between))))
    else:
        pooled_std = float("nan")
    effect = diff / pooled_std if np.isfinite(pooled_std) and pooled_std > 0 else float("nan")

    if num_perm <= 0 or not (within.size and between.size):
        return within_mean, between_mean, diff, effect, float("nan")

    fam_labels = [type_to_family[t] for t in type_list]
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(num_perm):
        perm = list(fam_labels)
        rng.shuffle(perm)
        perm_map = {t: perm[i] for i, t in enumerate(type_list)}
        w_p, b_p = _within_between(type_list, perm_map, mat)
        if not (w_p.size and b_p.size):
            continue
        diff_p = float(np.mean(b_p) - np.mean(w_p))
        if diff_p >= diff:
            count += 1
    pval = (count + 1) / (num_perm + 1)
    return within_mean, between_mean, diff, effect, float(pval)


def _centroid_acc(type_list: Sequence[str], type_to_family: Dict[str, str], proto: Dict[str, np.ndarray], metric_name: str) -> float:
    fam_to_types: Dict[str, List[str]] = {}
    for t in type_list:
        fam_to_types.setdefault(type_to_family[t], []).append(t)
    correct = 0
    total = 0
    for t in type_list:
        true_fam = type_to_family[t]
        best_fam = None
        best_dist = float("inf")
        for fam, members in fam_to_types.items():
            vecs = [proto[m] for m in members if not (fam == true_fam and m == t)]
            if not vecs:
                continue
            centroid = np.mean(np.stack(vecs, axis=0), axis=0)
            dist = _vec_distance(proto[t], centroid, metric_name)
            if dist < best_dist:
                best_dist = dist
                best_fam = fam
        if best_fam is None:
            continue
        total += 1
        if best_fam == true_fam:
            correct += 1
    return float(correct / total) if total else float("nan")


def _knn_purity(type_list: Sequence[str], type_to_family: Dict[str, str], mat: np.ndarray, k: int) -> float:
    total = 0.0
    count = 0
    for i, t in enumerate(type_list):
        dists = [(j, float(mat[i, j])) for j in range(len(type_list)) if j != i]
        dists.sort(key=lambda x: x[1])
        neighbors = [type_list[j] for j, _ in dists[: max(1, k)]]
        if not neighbors:
            continue
        same = sum(1 for n in neighbors if type_to_family[n] == type_to_family[t])
        total += same / float(len(neighbors))
        count += 1
    return float(total / count) if count else float("nan")


def _silhouette_from_matrix(type_list: Sequence[str], type_to_family: Dict[str, str], mat: np.ndarray) -> float:
    fam_to_indices: Dict[str, List[int]] = {}
    for i, t in enumerate(type_list):
        fam_to_indices.setdefault(type_to_family[t], []).append(i)
    scores = []
    for i, t in enumerate(type_list):
        same_idx = [j for j in fam_to_indices[type_to_family[t]] if j != i]
        if not same_idx:
            continue
        a = float(np.mean([mat[i, j] for j in same_idx]))
        b_vals = []
        for fam, idxs in fam_to_indices.items():
            if fam == type_to_family[t]:
                continue
            b_vals.append(float(np.mean([mat[i, j] for j in idxs])))
        if not b_vals:
            continue
        b = min(b_vals)
        denom = max(a, b, 1e-12)
        scores.append((b - a) / denom)
    return float(np.mean(scores)) if scores else float("nan")


def _misfit_rate(type_list: Sequence[str], type_to_family: Dict[str, str], mat: np.ndarray) -> float:
    idx_map = {t: i for i, t in enumerate(type_list)}
    fam_to_types: Dict[str, List[str]] = {}
    for t in type_list:
        fam_to_types.setdefault(type_to_family[t], []).append(t)

    misfits = 0
    total = 0
    for t in type_list:
        i = idx_map[t]
        fam_means: Dict[str, float] = {}
        for fam, members in fam_to_types.items():
            vals = []
            for m in members:
                j = idx_map[m]
                if i == j:
                    continue
                vals.append(float(mat[i, j]))
            if vals:
                fam_means[fam] = float(np.mean(vals))
        if not fam_means:
            continue
        best_fam = min(fam_means, key=fam_means.get)
        total += 1
        if best_fam != type_to_family[t]:
            misfits += 1
    return float(misfits / total) if total else float("nan")


def _family_similarity_metrics(
    by_corr: Dict[str, np.ndarray],
    fam_of: Dict[str, str],
    metric_name: str,
    num_perm: int,
    seed: int,
    knn_ks: Sequence[int],
) -> Dict[str, float]:
    type_list = sorted([c for c in by_corr if len(by_corr[c]) > 0])
    if len(type_list) < 3:
        out = {
            "family_within_mean": float("nan"),
            "family_between_mean": float("nan"),
            "family_between_minus_within": float("nan"),
            "family_effect_size": float("nan"),
            "family_permutation_p_value": float("nan"),
            "family_num_within_pairs": float("nan"),
            "family_num_between_pairs": float("nan"),
            "family_centroid_acc": float("nan"),
            "family_silhouette": float("nan"),
            "family_misfit_rate": float("nan"),
        }
        for k in knn_ks:
            out[f"family_knn_purity_k{k}"] = float("nan")
        return out

    proto = {t: np.mean(by_corr[t], axis=0) for t in type_list}
    mat = _pairwise_type_matrix(type_list, proto, metric_name)
    within, between = _within_between(type_list, fam_of, mat)
    w_mean, b_mean, diff, effect, pval = _effect_and_pvalue(
        within=within,
        between=between,
        type_list=type_list,
        type_to_family=fam_of,
        mat=mat,
        num_perm=num_perm,
        seed=seed,
    )

    out = {
        "family_within_mean": w_mean,
        "family_between_mean": b_mean,
        "family_between_minus_within": diff,
        "family_effect_size": effect,
        "family_permutation_p_value": pval,
        "family_num_within_pairs": float(within.size),
        "family_num_between_pairs": float(between.size),
        "family_centroid_acc": _centroid_acc(type_list, fam_of, proto, metric_name),
        "family_silhouette": _silhouette_from_matrix(type_list, fam_of, mat),
        "family_misfit_rate": _misfit_rate(type_list, fam_of, mat),
    }
    for k in knn_ks:
        out[f"family_knn_purity_k{k}"] = _knn_purity(type_list, fam_of, mat, k)
    return out


def _sequence_prefix(path: Path) -> str:
    stem = path.stem
    if "__" in stem:
        return stem.split("__", 1)[0]
    return stem.split("_", 1)[0]


def _discover_lidar_streams(data_root: Path) -> Dict[str, List[Path]]:
    lidar_dir = data_root / "samples" / "LIDAR_TOP"
    if not lidar_dir.exists():
        return {}
    streams: Dict[str, List[Path]] = {}
    for p in sorted(lidar_dir.glob("*.bin")):
        sid = _sequence_prefix(p)
        streams.setdefault(sid, []).append(p)
    return streams


def _sample_episodes(
    streams: Dict[str, List[Path]],
    episode_len: int,
    episode_stride: int,
    num_episodes: int,
    rng: np.random.Generator,
) -> List[List[Path]]:
    candidates: List[Tuple[str, int]] = []
    stride = max(1, episode_stride)
    for sid, paths in streams.items():
        if len(paths) < episode_len:
            continue
        for start in range(0, len(paths) - episode_len + 1, stride):
            candidates.append((sid, start))
    if not candidates:
        return []
    rng.shuffle(candidates)
    selected = candidates[: min(len(candidates), max(1, num_episodes))]
    return [streams[sid][start : start + episode_len] for sid, start in selected]


def _load_pointcloud(path: Path, max_points: int, rng: np.random.Generator) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.float32)
    if raw.size < 16:
        return np.zeros((0, 5), dtype=np.float32)
    if raw.size % 5 == 0:
        pts = raw.reshape(-1, 5)
    elif raw.size % 4 == 0:
        pts4 = raw.reshape(-1, 4)
        pad = np.zeros((pts4.shape[0], 1), dtype=np.float32)
        pts = np.concatenate([pts4, pad], axis=1)
    else:
        m = (raw.size // 5) * 5
        if m < 10:
            return np.zeros((0, 5), dtype=np.float32)
        pts = raw[:m].reshape(-1, 5)
    if len(pts) > max_points:
        idx = rng.choice(len(pts), size=max_points, replace=False)
        pts = pts[idx]
    return pts.astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AV LiDAR taxonomy metrics")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--encoders", type=str, required=True)
    p.add_argument("--corruptions", type=str, default="all")
    p.add_argument("--severities", type=str, default="1,2,3,4,5")
    p.add_argument("--distance", type=str, default="cosine", choices=["cosine", "l2"])
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--delta_variant", type=str, default="raw")
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--validation_severities", type=str, default="")
    p.add_argument("--max_samples_per_scope", type=int, default=256)
    p.add_argument("--max_points_per_cloud", type=int, default=4096)
    p.add_argument("--max_pairs", type=int, default=500000)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--episode_len", type=int, default=5)
    p.add_argument("--clean_len", type=int, default=0)
    p.add_argument("--corrupt_len", type=int, default=5)
    p.add_argument("--corrupt_full_episode", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--episode_stride", type=int, default=2)
    p.add_argument("--num_episodes", type=int, default=0)
    p.add_argument("--knn_ks", type=str, default="1,3,5")
    p.add_argument("--rep_mode", type=str, default="both", choices=["tokens", "pooled", "both"])

    # accepted for orchestrator compatibility
    p.add_argument("--taxonomy_mode", type=str, default="discover_then_validate")
    p.add_argument("--discovery_severities", type=str, default="")
    p.add_argument("--evaluate_on", type=str, default="validation")
    p.add_argument("--cluster_method", type=str, default="kmeans")
    p.add_argument("--n_discovery_families", type=int, default=0)
    p.add_argument("--k_min", type=int, default=2)
    p.add_argument("--k_max", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--delta_aggregation", type=str, default="trimmed_mean")
    p.add_argument("--trim_alpha", type=float, default=0.1)
    p.add_argument("--num_perm", type=int, default=1000)
    p.add_argument("--strict_index_check", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if args.episode_len < 1:
        raise ValueError("episode_len must be >= 1")
    if args.corrupt_full_episode:
        window_start = 0
        window_end = args.episode_len
    else:
        if args.episode_len < args.clean_len + args.corrupt_len:
            raise ValueError("episode_len must be >= clean_len + corrupt_len")
        window_start = args.clean_len
        window_end = args.clean_len + args.corrupt_len
    if window_end <= window_start:
        raise ValueError("Invalid temporal window bounds.")

    data_root = Path(args.data_root)
    encoders = _parse_csv_list(args.encoders)
    reps = ["tokens", "pooled"] if args.rep_mode == "both" else [args.rep_mode]
    corrs = ALL_LIDAR_CORRUPTIONS if args.corruptions == "all" else _parse_csv_list(args.corruptions)
    corrs = [c for c in corrs if c in FAMILY_OF]
    if not corrs:
        raise ValueError("No valid LiDAR corruptions selected.")

    knn_ks = _parse_int_list(args.knn_ks)
    knn_ks = [k for k in knn_ks if k > 0]
    if not knn_ks:
        knn_ks = [1, 3, 5]

    sev_eval = _parse_int_list(args.validation_severities) if args.validation_severities else _parse_int_list(args.severities)
    if not sev_eval:
        raise ValueError("No severities selected.")

    streams = _discover_lidar_streams(data_root)
    if not streams:
        raise FileNotFoundError(f"No LiDAR pointclouds found under: {data_root / 'samples' / 'LIDAR_TOP'}")

    n_target = args.num_episodes if args.num_episodes > 0 else max(8, args.max_samples_per_scope)
    episodes = _sample_episodes(
        streams=streams,
        episode_len=args.episode_len,
        episode_stride=args.episode_stride,
        num_episodes=n_target,
        rng=rng,
    )
    if not episodes:
        raise RuntimeError("No valid LiDAR temporal episodes found.")

    unique_paths = sorted({p for ep in episodes for p in ep})
    cloud_cache: Dict[Path, np.ndarray] = {}
    for p in unique_paths:
        pc = _load_pointcloud(p, max_points=args.max_points_per_cloud, rng=rng)
        if len(pc) >= 32:
            cloud_cache[p] = pc

    valid_episodes = [ep for ep in episodes if all(p in cloud_cache for p in ep)]
    if len(valid_episodes) < 8:
        raise RuntimeError("Too few valid LiDAR episodes after load filtering.")

    window_clouds = [[cloud_cache[p] for p in ep[window_start:window_end]] for ep in valid_episodes]
    clean_window_base = np.stack(
        [np.mean(np.stack([_lidar_base_feature(pc) for pc in win], axis=0), axis=0) for win in window_clouds],
        axis=0,
    )

    run_dir = Path(args.output_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: List[Dict[str, object]] = []
    detail = {
        "modality": "lidar",
        "encoders": encoders,
        "representations": reps,
        "corruptions": corrs,
        "severities_eval": sev_eval,
        "distance": args.distance,
        "n_episodes": len(valid_episodes),
        "episode_len": args.episode_len,
        "clean_len": args.clean_len,
        "corrupt_len": args.corrupt_len,
        "corrupt_full_episode": bool(args.corrupt_full_episode),
        "window_start": window_start,
        "window_end": window_end,
        "episode_stride": args.episode_stride,
        "knn_ks": knn_ks,
    }

    for sev in sev_eval:
        corr_window_base: Dict[str, np.ndarray] = {}
        for corr in corrs:
            corr_rows = []
            for ep_idx, win in enumerate(window_clouds):
                rng_ep = np.random.default_rng(args.seed + sev * 100003 + ep_idx * 1543 + _encoder_seed(corr))
                corr_feats = []
                for pc in win:
                    cpc = _apply_lidar_corruption(pc, corr, sev, rng=rng_ep)
                    corr_feats.append(_lidar_base_feature(cpc))
                corr_rows.append(np.mean(np.stack(corr_feats, axis=0), axis=0))
            corr_window_base[corr] = np.stack(corr_rows, axis=0)

        for encoder in encoders:
            for rep in reps:
                clean_feat = _build_representation(clean_window_base, encoder=encoder, rep_mode=rep)
                by_corr: Dict[str, np.ndarray] = {}
                for corr in corrs:
                    corr_feat = _build_representation(corr_window_base[corr], encoder=encoder, rep_mode=rep)
                    by_corr[corr] = (corr_feat - clean_feat).astype(np.float32)

                x_all = np.concatenate([by_corr[c] for c in corrs], axis=0)
                y_all: List[str] = []
                for c in corrs:
                    y_all.extend([FAMILY_OF[c]] * len(by_corr[c]))

                global_metrics = _evaluate_space(
                    x=x_all,
                    labels=y_all,
                    metric_name=args.distance,
                    max_pairs=args.max_pairs,
                    seed=args.seed + sev,
                )
                family_metrics = _family_similarity_metrics(
                    by_corr=by_corr,
                    fam_of=FAMILY_OF,
                    metric_name=args.distance,
                    num_perm=args.num_perm,
                    seed=args.seed + sev + _encoder_seed(f"{encoder}::{rep}"),
                    knn_ks=knn_ks,
                )

                for mk, mv in {**global_metrics, **family_metrics}.items():
                    metric_rows.append(
                        {
                            "encoder": encoder,
                            "representation": rep,
                            "delta_variant": args.delta_variant,
                            "scope": "corruption:all",
                            "distance": "n/a" if mk.startswith("ari_") or mk.startswith("nmi_") else args.distance,
                            "metric": mk,
                            "value": float(mv),
                        }
                    )

                for corr in corrs:
                    x_scope, y_scope = _scope_subset(
                        corr_name=corr,
                        by_corr=by_corr,
                        fam_of=FAMILY_OF,
                        rng=rng,
                        cap_bg_per_family=max(16, len(by_corr[corr]) // 2),
                    )
                    m_scope = _evaluate_space(
                        x=x_scope,
                        labels=y_scope,
                        metric_name=args.distance,
                        max_pairs=args.max_pairs,
                        seed=args.seed + sev + _encoder_seed(corr),
                    )
                    for mk, mv in m_scope.items():
                        metric_rows.append(
                            {
                                "encoder": encoder,
                                "representation": rep,
                                "delta_variant": args.delta_variant,
                                "scope": f"corruption:{corr}",
                                "distance": "n/a" if mk.startswith("ari_") or mk.startswith("nmi_") else args.distance,
                                "metric": mk,
                                "value": float(mv),
                            }
                        )

    _write_csv(run_dir / "metrics_table.csv", metric_rows)
    with (run_dir / "run_config.json").open("w") as f:
        json.dump(detail, f, indent=2)
    print(f"[DONE] lidar metrics written: {run_dir / 'metrics_table.csv'}")
    print(f"[DONE] rows: {len(metric_rows)}")


if __name__ == "__main__":
    main()
