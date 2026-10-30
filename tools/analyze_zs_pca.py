#!/usr/bin/env python3
"""analyze_zs_pca.py — 240M 基线（降维训练之前）输出 token 的按位置 PCA 冗余分析

触发背景（2026-10-30）：编码器末端加 FC 把 z_s 每 token 从 1248 维压到 448/224 维、
全量微调后 PSNR 只掉 0.25/1.85 dB（base 27.936 → zs448 27.688 → zs224 26.083 dB，
5,000 张 IN-100 val）。用户由此提出「原始 token 的 PCA 主成分很可能是冗余的」。
本脚本对**降维训练之前的基线模型**在测试集上输出的 token 做按位置 PCA，量化冗余度，
并保留 token 向量/基，供后续实验（T3 零训练 PCA 截断→解码器）。

数据：/userdata/dsh-offload/assocvtc_zpca/tokens/zs_base_m240_t16_fp16.npz
    z_s [N,32,1248] fp16（N=5000）、z_cls [N,1,1248] fp16、labels、paths

产物（--out_dir，默认 /home/linaro/dsh/_assocvtc_zpca/analysis/）：
    pca_perpos_spectra.json     每位置 λ 谱（抽样点）、累计方差、rank@90/95/99/99.9、
                                participation ratio、前 224/448/624 维累计方差、不居中口径、
                                split-half 稳定性
    pca_perpos_components.npz   每位置均值 (32,1248) + 前 64 主成分 (32,64,1248) float32
                                + 完整中心化特征值 (32,1248) float64
    crosspos.json               位置均值余弦/范数、主角度与子空间重叠矩阵、池化 PCA、
                                跨位置（每图 32×1248）有效秩、z_cls 同套 PCA
    run_meta.json               输入 sha256、shape/dtype、版本、耗时、自查结果
图（--fig_dir，默认 仓库/doc/2026-10-30/artifacts/zpca/，png+pdf）：
    fig_pca_spectrum, fig_rank_at_threshold, fig_crosspos_angles, fig_var_at_448_224

复跑（数据到位后）：
    cd /home/linaro/dsh/AssocVTC
    PYTHONPATH=$PWD python tools/analyze_zs_pca.py \
        --tokens /userdata/dsh-offload/assocvtc_zpca/tokens/zs_base_m240_t16_fp16.npz \
        --out_dir /home/linaro/dsh/_assocvtc_zpca/analysis \
        --fig_dir doc/2026-10-30/artifacts/zpca
数据未到位时可用合成数据把流程跑通：
    PYTHONPATH=$PWD python tools/analyze_zs_pca.py --synthetic 5000 \
        --out_dir /tmp/zpca_syn --fig_dir /tmp/zpca_syn/figs --no_split_half
仅重画图（读已有 json）：
    PYTHONPATH=$PWD python tools/analyze_zs_pca.py --figs_only \
        --out_dir ... --fig_dir ...
"""
import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone

import numpy as np

# ----------------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------------
THRESHOLDS = (0.90, 0.95, 0.99, 0.999)
TOP_KS = (1, 8, 16, 32, 64, 128, 224, 448, 624, 896, 1120)   # 累计方差抽样点
SAMPLE_IDX = tuple(range(1, 17)) + (24, 32, 48, 64, 96, 128, 160, 192, 224, 256,
                                   320, 448, 512, 624, 768, 896, 1024, 1248)
RECON_RS = (112, 224, 448, 624, 896)     # 报告 token 空间相对 L2 误差的点

# 既有对照数字（不重跑；取自 doc/2026-10-26/REPORT_rfid_m240_pca.md §11/§12）
REFERENCE = {
    "psnr_db_5k": {"base_m240_t16": 27.936, "zs448_arm": 27.688, "zs224_arm": 26.083},
    "ssim_5k": {"base_m240_t16": 0.8288, "zs448_arm": 0.8164, "zs224_arm": 0.7508},
    "l1_255_5k": {"base_m240_t16": 7.004, "zs448_arm": 7.265, "zs224_arm": 8.707},
    "note": ("每图 39,936 值（32×1248）→ zs448: 14,336 值（32×448）→ zs224: 7,168 值（32×224）；"
             "warm start 是 decoder.mem_proj.weight 的 SVD 秩近似，不是数据 PCA"),
    "source": "doc/2026-10-26/REPORT_rfid_m240_pca.md §11/§12",
}

DEFAULT_TOKENS = "/userdata/dsh-offload/assocvtc_zpca/tokens/zs_base_m240_t16_fp16.npz"
DEFAULT_OUT = "/home/linaro/dsh/_assocvtc_zpca/analysis"


# ----------------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------------
def _sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _eigh_desc(C):
    """对称矩阵特征分解，特征值/特征向量按特征值降序返回（float64）。"""
    lam, U = np.linalg.eigh(0.5 * (C + C.T))
    return lam[::-1].copy(), U[:, ::-1].copy()


def _fix_signs(U):
    """主成分符号确定性：每列绝对值最大分量为正，保证可复跑同一 npz。"""
    idx = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[idx, np.arange(U.shape[1])])
    signs[signs == 0] = 1.0
    return U * signs


def _stats(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
    }


def _samples(values, idxs, offset=0):
    """从 1-based 下标元组取抽样点，键为 1-based 秩（超出维数的点自动跳过）。"""
    n = len(values)
    return {str(int(i)): float(values[i - 1 - offset]) for i in idxs
            if 1 <= i - offset <= n}


def _cum_rank(cum, thr):
    """累计方差首次 >= thr 所需的主成分个数（1-based，最大 = D）。"""
    r = int(np.searchsorted(cum, thr, side="left")) + 1
    return min(r, len(cum))


def _participation_ratio(lam):
    lam = np.clip(np.asarray(lam, dtype=np.float64), 0.0, None)
    s = lam.sum()
    if s <= 0:
        return float("nan")
    return float(s * s / (lam * lam).sum())


def _entropy_rank(lam):
    lam = np.clip(np.asarray(lam, dtype=np.float64), 0.0, None)
    s = lam.sum()
    if s <= 0:
        return float("nan")
    p = lam[lam > 0] / s
    return float(np.exp(-(p * np.log(p)).sum()))


def _print(*a, **k):
    print(*a, **k, flush=True)


# ----------------------------------------------------------------------------
# 单个矩阵的 PCA 谱（z_cls 用；positions 在循环里内联算，便于同时拿 Gram）
# ----------------------------------------------------------------------------
def analyze_matrix(Z, split_half=False, rng=None, top_r_basis=224):
    """Z: [n, D] float64。返回谱统计与（可选）基。"""
    n, D = Z.shape
    mu = Z.mean(axis=0)
    G = Z.T @ Z
    G = 0.5 * (G + G.T)
    C = (G - n * np.outer(mu, mu)) / (n - 1)
    lam_c, U_c = _eigh_desc(C)
    U_c = _fix_signs(U_c)
    lam_u = np.linalg.eigvalsh(G / n)[::-1].copy()   # 不居中（二阶矩）PCA

    out = _spectrum_dict(lam_c, lam_u, mu, G, n)
    out["_basis"] = U_c[:, :top_r_basis]
    if split_half:
        assert rng is not None
        perm = rng.permutation(n)
        a, b = perm[: n // 2], perm[n // 2:]
        sub = {}
        for tag, ix in (("fit_a", a), ("fit_b", b)):
            Za = Z[ix]
            mua = Za.mean(axis=0)
            Ca = (Za.T @ Za - len(ix) * np.outer(mua, mua)) / (len(ix) - 1)
            la, Ua = _eigh_desc(Ca)
            Ua = _fix_signs(Ua)
            cum = np.cumsum(np.clip(la, 0, None))
            cum = cum / cum[-1]
            sub[tag] = {
                "rank_at": {f"{t:.3f}": _cum_rank(cum, t) for t in THRESHOLDS},
                "participation_ratio": _participation_ratio(la),
                "eigenvalues": la,
                "_basis": Ua[:, :top_r_basis],
            }
        cosang = np.clip((sub["fit_a"]["_basis"].T @ sub["fit_b"]["_basis"]).sum(axis=0)
                         / top_r_basis, 0.0, 1.0)
        # 主角度：U_a^T U_b 的奇异值（= cos 的逐分量上界），这里报告均值角
        sv = np.linalg.svd(sub["fit_a"]["_basis"].T @ sub["fit_b"]["_basis"],
                           compute_uv=False)
        ang = np.degrees(np.arccos(np.clip(sv, 0.0, 1.0)))
        out["split_half"] = {
            "fit_a": {k: v for k, v in sub["fit_a"].items()
                      if k in ("rank_at", "participation_ratio")},
            "fit_b": {k: v for k, v in sub["fit_b"].items()
                      if k in ("rank_at", "participation_ratio")},
            "abs_rank_diff": {f"{t:.3f}": abs(sub["fit_a"]["rank_at"][f"{t:.3f}"]
                                              - sub["fit_b"]["rank_at"][f"{t:.3f}"])
                              for t in THRESHOLDS},
            "mean_angle_deg_top224": float(ang.mean()),
            "max_angle_deg_top224": float(ang.max()),
            "top100_evr_l1_rel_diff": float(
                np.abs(_norm_evr(sub["fit_a"]["eigenvalues"], 100)
                       - _norm_evr(sub["fit_b"]["eigenvalues"], 100)).sum()),
        }
    return out


def _norm_evr(lam, k):
    lam = np.clip(np.asarray(lam, dtype=np.float64), 0.0, None)[:k]
    return lam / lam.sum()


def _spectrum_dict(lam_c, lam_u, mu, G, n):
    """主导谱统计（中心化口径为主，不居中口径并列）。"""
    D = len(lam_c)
    lam_c_raw = lam_c
    lam_c = np.clip(lam_c, 0.0, None)
    total_c = float(lam_c.sum())
    cum = np.cumsum(lam_c) / total_c
    cum = np.minimum(cum, 1.0)
    lam_u = np.clip(np.asarray(lam_u, dtype=np.float64), 0.0, None)
    total_u = float(lam_u.sum())
    cum_u = np.cumsum(lam_u) / total_u
    cum_u = np.minimum(cum_u, 1.0)

    mean_sq = float(np.trace(G) / n)          # E||z||^2（原始能量）
    mean_energy_frac = float((mu @ mu) / mean_sq) if mean_sq > 0 else float("nan")

    d = {
        "mean_norm": float(np.linalg.norm(mu)),
        "mean_sq_norm": mean_sq,
        "total_var_centered": total_c,
        "total_energy_raw": total_u,
        "mean_energy_frac": mean_energy_frac,
        "eigen_min_centered": float(np.min(lam_c_raw)),
        "eigenvalues_centered_sample": _samples(lam_c, SAMPLE_IDX),
        "evr_sample": _samples(lam_c / total_c, SAMPLE_IDX),
        "cum_var_sample": _samples(cum, SAMPLE_IDX),
        "rank_at": {f"{t:.3f}": _cum_rank(cum, t) for t in THRESHOLDS},
        "rank_at_frac": {f"{t:.3f}": _cum_rank(cum, t) / D for t in THRESHOLDS},
        "participation_ratio": _participation_ratio(lam_c),
        "entropy_rank": _entropy_rank(lam_c),
        "top_var": {str(k): float(cum[k - 1]) for k in TOP_KS if k <= D},
        "recon_rel_l2_at": {str(r): float(np.sqrt(max(1.0 - cum[r - 1], 0.0)))
                            for r in RECON_RS if r < D},
        "uncentered": {
            "rank_at": {f"{t:.3f}": _cum_rank(cum_u, t) for t in THRESHOLDS},
            "participation_ratio": _participation_ratio(lam_u),
            "entropy_rank": _entropy_rank(lam_u),
            "top_var": {str(k): float(cum_u[k - 1]) for k in TOP_KS if k <= D},
            "top1_share": float(cum_u[0]),
            "eigenvalues_sample": _samples(lam_u, SAMPLE_IDX),
        },
    }
    return d


# ----------------------------------------------------------------------------
# 数据载入
# ----------------------------------------------------------------------------
def load_tokens(path):
    with np.load(path) as f:
        keys = list(f.keys())
        z_s = np.asarray(f["z_s"][...])
        z_cls = np.asarray(f["z_cls"][...]) if "z_cls" in keys else None
        labels = np.asarray(f["labels"][...]) if "labels" in keys else None
        paths = np.asarray(f["paths"][...]) if "paths" in keys else None
    if z_s.ndim != 3:
        raise SystemExit(f"z_s 维度应为 [N,K,D]，实际 {z_s.shape}")
    return z_s, _squeeze_cls(z_cls), labels, paths, keys


def _squeeze_cls(z_cls):
    if z_cls is not None and z_cls.ndim == 3 and z_cls.shape[1] == 1:
        return z_cls[:, 0, :]
    return z_cls


def make_synthetic(n, K=32, D=1248, seed=0):
    """合成数据：位置均值 + 64 维共享子空间 + 各向同性残差，fp16 存储模拟真数据。"""
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal((K, D)) * 0.35
    shared = rng.standard_normal((64, D)) / np.sqrt(D)
    coef = rng.standard_normal((n, K, 64)) * 1.2
    resid = rng.standard_normal((n, K, D)) * 0.55
    z_s = (mu[None] + coef @ shared + resid).astype(np.float16)
    mu_cls = rng.standard_normal(D) * 0.3
    z_cls = (mu_cls[None] + (rng.standard_normal((n, 96)) * 1.1) @
             (rng.standard_normal((96, D)) / np.sqrt(D)) +
             rng.standard_normal((n, D)) * 0.5).astype(np.float16)[:, None, :]
    labels = rng.integers(0, 100, size=n).astype(np.int16)
    paths = np.array([f"synthetic/{i:05d}.JPEG" for i in range(n)])
    return z_s, z_cls, labels, paths, ["z_s", "z_cls", "labels", "paths"]


# ----------------------------------------------------------------------------
# 主分析：逐位置 PCA + 跨位置结构
# ----------------------------------------------------------------------------
def main_analysis(z_s, z_cls, out_dir, split_half=True, seed=0, top_components=64):
    t0 = time.time()
    N, K, D = z_s.shape
    _print(f"[data] z_s shape={z_s.shape} dtype={z_s.dtype} "
           f"nan={int(np.isnan(z_s).sum())} inf={int(np.isinf(z_s).sum())}")
    rng = np.random.default_rng(seed)

    pos_records = []
    pos_means = np.empty((K, D), dtype=np.float64)
    pos_top = np.empty((K, top_components, D), dtype=np.float64)
    eig_full = np.empty((K, D), dtype=np.float64)
    # 池化累计量
    sum_G = np.zeros((D, D), dtype=np.float64)              # Σ_k G_k
    sum_Cpp = np.zeros((D, D), dtype=np.float64)            # Σ_k (G_k - N μμ^T)/N
    U224 = np.empty((K, D, 224), dtype=np.float64)          # 每位置 top-224 基（主角度用）
    checks = {"cum_monotone": True, "eigsum_vs_trace_max_rel_err": 0.0,
              "evr_sum_max_abs_dev": 0.0, "split_half_enabled": bool(split_half)}

    for k in range(K):
        tk = time.time()
        Z = z_s[:, k, :].astype(np.float64)          # [N,D] 只此一份 float64
        n = Z.shape[0]
        mu = Z.mean(axis=0)
        G = Z.T @ Z
        G = 0.5 * (G + G.T)
        # 中心化协方差
        C = (G - n * np.outer(mu, mu)) / (n - 1)
        lam_c, U_c = _eigh_desc(C)
        lam_c_raw = lam_c
        lam_c = np.clip(lam_c, 0.0, None)
        U_c = _fix_signs(U_c)
        lam_u = np.linalg.eigvalsh(G / n)[::-1].copy()

        # 累计方差自查
        total = float(lam_c.sum())
        cum = np.cumsum(lam_c) / total
        checks["cum_monotone"] &= bool(np.all(np.diff(cum) >= -1e-12))
        evr_sum_dev = float(abs(cum[-1] - 1.0))
        checks["evr_sum_max_abs_dev"] = max(checks["evr_sum_max_abs_dev"], evr_sum_dev)
        trace_ref = float((np.trace(G) - n * mu @ mu) / (n - 1))
        rel = float(abs(total - trace_ref) / max(abs(trace_ref), 1e-300))
        checks["eigsum_vs_trace_max_rel_err"] = max(
            checks["eigsum_vs_trace_max_rel_err"], rel)

        rec = _spectrum_dict(lam_c_raw, lam_u, mu, G, n)
        rec["pos"] = k
        pos_means[k] = mu
        pos_top[k] = U_c[:, :top_components].T
        eig_full[k] = lam_c_raw
        U224[k] = U_c[:, :224]

        sum_G += G
        sum_Cpp += (G - n * np.outer(mu, mu)) / n

        if split_half:
            sub = analyze_matrix(Z, split_half=True, rng=rng, top_r_basis=224)
            rec["split_half"] = sub["split_half"]
            del sub

        pos_records.append(rec)
        _print(f"[pos {k:02d}/{K - 1}] mu_norm={rec['mean_norm']:.3f} "
               f"rank@90/95/99/99.9="
               f"{'/'.join(str(rec['rank_at'][f'{t:.3f}']) for t in THRESHOLDS)} "
               f"PR={rec['participation_ratio']:.1f} "
               f"top448={rec['top_var']['448']:.4f} top224={rec['top_var']['224']:.4f} "
               f"({time.time() - tk:.1f}s)")

    # ---------------- 逐位置汇总 ----------------
    agg = {}
    for key, path in (
        ("rank_at", lambda r: r["rank_at"]),
        ("top_var", lambda r: r["top_var"]),
    ):
        pass
    agg["rank_at"] = {f"{t:.3f}": _stats([r["rank_at"][f"{t:.3f}"] for r in pos_records])
                      for t in THRESHOLDS}
    agg["rank_at_frac"] = {f"{t:.3f}": _stats([r["rank_at_frac"][f"{t:.3f}"] for r in pos_records])
                           for t in THRESHOLDS}
    agg["top_var"] = {str(k): _stats([r["top_var"][str(k)] for r in pos_records])
                      for k in TOP_KS if str(k) in pos_records[0]["top_var"]}
    agg["participation_ratio"] = _stats([r["participation_ratio"] for r in pos_records])
    agg["entropy_rank"] = _stats([r["entropy_rank"] for r in pos_records])
    agg["mean_norm"] = _stats([r["mean_norm"] for r in pos_records])
    agg["mean_sq_norm"] = _stats([r["mean_sq_norm"] for r in pos_records])
    agg["mean_energy_frac"] = _stats([r["mean_energy_frac"] for r in pos_records])
    agg["total_var_centered"] = _stats([r["total_var_centered"] for r in pos_records])
    agg["uncentered"] = {
        "rank_at": {f"{t:.3f}": _stats([r["uncentered"]["rank_at"][f"{t:.3f}"]
                                        for r in pos_records]) for t in THRESHOLDS},
        "top_var": {str(k): _stats([r["uncentered"]["top_var"][str(k)] for r in pos_records])
                    for k in TOP_KS if str(k) in pos_records[0]["uncentered"]["top_var"]},
        "participation_ratio": _stats([r["uncentered"]["participation_ratio"]
                                       for r in pos_records]),
        "top1_share": _stats([r["uncentered"]["top1_share"] for r in pos_records]),
    }
    if split_half:
        agg["split_half"] = {
            "abs_rank_diff": {f"{t:.3f}": _stats([r["split_half"]["abs_rank_diff"][f"{t:.3f}"]
                                                  for r in pos_records])
                              for t in THRESHOLDS},
            "mean_angle_deg_top224": _stats([r["split_half"]["mean_angle_deg_top224"]
                                             for r in pos_records]),
            "max_angle_deg_top224": _stats([r["split_half"]["max_angle_deg_top224"]
                                            for r in pos_records]),
            "top100_evr_l1_rel_diff": _stats([r["split_half"]["top100_evr_l1_rel_diff"]
                                              for r in pos_records]),
        }

    # ---------------- 跨位置结构 ----------------
    _print("[cross-pos] 位置均值 / 主角度 / 池化 PCA / 每图跨位置秩 ...")
    mu_norm = np.linalg.norm(pos_means, axis=1)
    mu_hat = pos_means / np.clip(mu_norm[:, None], 1e-12, None)
    cos_mu = mu_hat @ mu_hat.T
    iu = np.triu_indices(K, 1)
    crosspos = {
        "position_means": {
            "norms": [float(x) for x in mu_norm],
            "cos_matrix": [[float(x) for x in row] for row in cos_mu],
            "pairwise_cos": _stats(cos_mu[iu]),
            "frac_pairwise_cos_gt_0.9": float(np.mean(cos_mu[iu] > 0.9)),
            "frac_pairwise_cos_gt_0.5": float(np.mean(cos_mu[iu] > 0.5)),
            "argmin_pair": [int(iu[0][np.argmin(cos_mu[iu])]),
                            int(iu[1][np.argmin(cos_mu[iu])])],
            "argmax_pair": [int(iu[0][np.argmax(cos_mu[iu])]),
                            int(iu[1][np.argmax(cos_mu[iu])])],
        },
        "principal_angles": {},
        "pooled": {},
    }
    for r in (32, 64, 224):
        angle_mat = np.full((K, K), np.nan)       # 每对子空间的平均主角度
        maxang_mat = np.full((K, K), np.nan)      # 每对子空间的最大主角度
        overlap_mat = np.full((K, K), np.nan)
        np.fill_diagonal(angle_mat, 0.0)
        np.fill_diagonal(maxang_mat, 0.0)
        np.fill_diagonal(overlap_mat, 1.0)
        for i in range(K):
            Ui = U224[i, :, :r]
            for j in range(i + 1, K):
                M = Ui.T @ U224[j, :, :r]
                sv = np.linalg.svd(M, compute_uv=False)
                sv = np.clip(sv, 0.0, 1.0)
                ang = np.degrees(np.arccos(sv))
                angle_mat[i, j] = angle_mat[j, i] = float(ang.mean())
                maxang_mat[i, j] = maxang_mat[j, i] = float(ang.max())
                overlap_mat[i, j] = overlap_mat[j, i] = float((sv * sv).sum() / r)
        ti = np.triu_indices(K, 1)
        ai, mai, oi = angle_mat[ti], maxang_mat[ti], overlap_mat[ti]
        entry = {
            "summary": {
                "mean_angle_deg": _stats(ai),
                "max_angle_deg": _stats(mai),
                "overlap": _stats(oi),
                "random_overlap_expect_r_over_D": float(r / D),
                "random_mean_angle_deg_approx": float(
                    np.degrees(np.arccos(np.sqrt(r / D)))),
                "frac_pairs_overlap_gt_0.5": float(np.mean(oi > 0.5)),
                "frac_pairs_overlap_gt_0.8": float(np.mean(oi > 0.8)),
                "best_aligned_pair_min_angle": [
                    int(ti[0][np.argmin(ai)]), int(ti[1][np.argmin(ai)])],
                "worst_aligned_pair_max_angle": [
                    int(ti[0][np.argmax(mai)]), int(ti[1][np.argmax(mai)])],
            }
        }
        if r in (32, 224):
            entry["mean_angle_deg_matrix"] = [[float(x) for x in row] for row in angle_mat]
            entry["max_angle_deg_matrix"] = [[float(x) for x in row] for row in maxang_mat]
            entry["overlap_matrix"] = [[float(x) for x in row] for row in overlap_mat]
        crosspos["principal_angles"][str(r)] = entry

    # 池化 PCA
    Ntot = N * K
    M_raw = sum_G / Ntot
    g = pos_means.mean(axis=0)
    M_gc = M_raw - np.outer(g, g)
    # sum_Cpp 存的是 Σ_k (G_k - N μμ^T)/N = Σ_k 每位置总体协方差，故除以 K
    C_pp = sum_Cpp / K                # 等价于 (1/K)Σ_k 每位置总体协方差

    def pooled_block(mat, kind):
        lam = np.linalg.eigvalsh(0.5 * (mat + mat.T))[::-1].copy()
        lam = np.clip(lam, 0.0, None)
        cum = np.cumsum(lam) / lam.sum()
        cum = np.minimum(cum, 1.0)
        blk = {
            "kind": kind,
            "top_var": {str(k): float(cum[k - 1]) for k in TOP_KS if k <= D},
            "rank_at": {f"{t:.3f}": _cum_rank(cum, t) for t in THRESHOLDS},
            "participation_ratio": _participation_ratio(lam),
            "entropy_rank": _entropy_rank(lam),
            "eigenvalues_sample": _samples(lam, SAMPLE_IDX),
            "cum_var_sample": _samples(cum, SAMPLE_IDX),
            "total_variance": float(lam.sum()),
        }
        if kind == "global_centered" or kind == "perpos_centered":
            blk["basis"] = _fix_signs(_eigh_desc(mat)[1][:, :224])
        elif kind == "raw":
            blk["basis"] = _fix_signs(_eigh_desc(mat)[1][:, :224])
        return blk, lam

    raw_blk, _ = pooled_block(M_raw, "raw")
    gc_blk, _ = pooled_block(M_gc, "global_centered")
    pp_blk, _ = pooled_block(C_pp, "perpos_centered")
    crosspos["pooled"] = {
        "raw": raw_blk,
        "global_centered": gc_blk,
        "perpos_centered": pp_blk,
        "definitions": {
            "raw": "160,000 个 token（5000×32）不居中，二阶矩 PCA；对应「一个共享线性投影最多保留的 token 原始能量」",
            "global_centered": "减去全部 token 的全局均值后 PCA（全局均值 = 位置均值的平均）",
            "perpos_centered": "每个 token 先减其位置均值再池化（= 各位置协方差的平均）",
        },
    }
    del raw_blk, gc_blk, pp_blk

    # 各位置 top-224 子空间 vs 池化逐位置居中 top-224（共享子空间）的主角度
    Upp = crosspos["pooled"]["perpos_centered"].pop("basis")
    Uraw = crosspos["pooled"]["raw"].pop("basis")
    Ugc = crosspos["pooled"]["global_centered"].pop("basis")
    shared_angle = []
    for i in range(K):
        sv = np.linalg.svd(U224[i].T @ Upp, compute_uv=False)
        shared_angle.append(float(np.degrees(np.arccos(np.clip(sv, 0, 1))).mean()))
    crosspos["shared_subspace"] = {
        "pooled_perpos_centered_top224": {
            "angle_to_positions_deg": shared_angle,
            "summary": _stats(shared_angle),
        },
        "raw_top224": {
            "angle_to_positions_deg": [
                float(np.degrees(np.arccos(np.clip(
                    np.linalg.svd(U224[i].T @ Uraw, compute_uv=False), 0, 1))).mean())
                for i in range(K)],
        },
        "global_centered_top224": {
            "angle_to_positions_deg": [
                float(np.degrees(np.arccos(np.clip(
                    np.linalg.svd(U224[i].T @ Ugc, compute_uv=False), 0, 1))).mean())
                for i in range(K)],
        },
    }
    crosspos["shared_subspace"]["raw_top224"]["summary"] = _stats(
        crosspos["shared_subspace"]["raw_top224"]["angle_to_positions_deg"])
    crosspos["shared_subspace"]["global_centered_top224"]["summary"] = _stats(
        crosspos["shared_subspace"]["global_centered_top224"]["angle_to_positions_deg"])

    # ---------------- 每图 32×1248 的跨位置有效秩 ----------------
    crosspos["per_image_crosspos"] = per_image_crosspos(z_s, pos_means)

    # ---------------- z_cls 同套 PCA ----------------
    zcls = {}
    if z_cls is not None:
        Zc = np.asarray(z_cls, dtype=np.float64)
        _print(f"[z_cls] shape={Zc.shape} 做 PCA ...")
        cls_ana = analyze_matrix(Zc, split_half=False)
        Ucls = cls_ana.pop("_basis")
        del cls_ana["uncentered"]["eigenvalues_sample"]
        zcls = cls_ana
        # z_cls 与 z_s 的关系
        cls_mean = Zc.mean(axis=0)
        cls_n = cls_mean / max(np.linalg.norm(cls_mean), 1e-12)
        cos_to_mu = pos_means @ cls_n / np.clip(mu_norm, 1e-12, None)
        ang_cls = []
        for i in range(K):
            sv = np.linalg.svd(Ucls.T @ U224[i], compute_uv=False)
            ang_cls.append(float(np.degrees(np.arccos(np.clip(sv, 0, 1))).mean()))
        crosspos["zcls_vs_zs"] = {
            "cos_zcls_mean_to_posmeans": [float(x) for x in cos_to_mu],
            "cos_summary": _stats(cos_to_mu),
            "angle_zcls_top224_to_pos_top224_deg": ang_cls,
            "angle_summary": _stats(ang_cls),
        }

    # ---------------- 自查 ----------------
    checks["cum_monotone"] = bool(checks["cum_monotone"])
    checks["n_positions"] = K
    checks["n_images"] = N
    checks["dim"] = D
    checks["split_half_enabled"] = bool(split_half)
    if split_half:
        checks["split_half_rank95_median_abs_diff"] = float(
            np.median([r["split_half"]["abs_rank_diff"]["0.950"] for r in pos_records]))
    # rank / top_var 单调性
    checks["rank_at_monotone_all"] = bool(all(
        r["rank_at"]["0.900"] <= r["rank_at"]["0.950"] <= r["rank_at"]["0.990"]
        <= r["rank_at"]["0.999"] for r in pos_records))
    checks["top_var_monotone_in_k"] = bool(all(
        r["top_var"][str(a)] <= r["top_var"][str(b)]
        for r in pos_records
        for a, b in zip(TOP_KS[:-1], TOP_KS[1:])))
    checks["uncentered_rank_at_monotone_all"] = bool(all(
        r["uncentered"]["rank_at"]["0.900"] <= r["uncentered"]["rank_at"]["0.950"]
        <= r["uncentered"]["rank_at"]["0.990"]
        <= r["uncentered"]["rank_at"]["0.999"] for r in pos_records))

    return {
        "positions": pos_records,
        "aggregate": agg,
        "crosspos": crosspos,
        "z_cls": zcls,
        "checks": checks,
        "pos_means": pos_means,
        "pos_top": pos_top,
        "eig_full": eig_full,
        "elapsed_compute_s": time.time() - t0,
    }


def per_image_crosspos(z_s, pos_means):
    """每张图 32 个 token（1248 维）跨位置秩；三个口径：原样 / 减位置均值 / 减图内均值。"""
    N, K, D = z_s.shape
    variants = ("raw", "minus_posmean", "minus_imgmean")
    acc = {v: {} for v in variants}
    chunk = 250
    metrics = ("pr", "entropy_rank", "rank90", "rank95", "rank99",
               "top1_share", "pairwise_cos_mean", "frac_cos_gt_0.9")
    for v in variants:
        for m in metrics:
            acc[v][m] = []
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        Zc = z_s[s:e].astype(np.float64)                    # [b,K,D]
        for v in variants:
            if v == "raw":
                A = Zc
            elif v == "minus_posmean":
                A = Zc - pos_means[None]
            else:
                A = Zc - Zc.mean(axis=1, keepdims=True)
            G = A @ A.transpose(0, 2, 1)                    # [b,K,K]
            s2 = np.linalg.eigvalsh(0.5 * (G + G.transpose(0, 2, 1)))[:, ::-1]
            s2 = np.clip(s2, 0.0, None)
            tot = s2.sum(axis=1, keepdims=True)
            tot = np.clip(tot, 1e-300, None)
            p = s2 / tot
            pr = tot[:, 0] ** 2 / np.clip((s2 * s2).sum(axis=1), 1e-300, None)
            ent = np.exp(-(p * np.log(np.clip(p, 1e-300, None))).sum(axis=1))
            cum = np.cumsum(p, axis=1)
            acc[v]["pr"].extend(pr.tolist())
            acc[v]["entropy_rank"].extend(ent.tolist())
            for t in (0.90, 0.95, 0.99):
                hit = cum >= t
                r = (np.argmax(hit, axis=1) + 1).astype(np.int64)
                r = np.where(hit.any(axis=1), r, K)
                acc[v][f"rank{int(t * 100)}"].extend(r.tolist())
            acc[v]["top1_share"].extend(p[:, 0].tolist())
            # 两两余弦（原样与减位置均值口径都有意义）
            An = A / np.clip(np.linalg.norm(A, axis=2, keepdims=True), 1e-12, None)
            cos = An @ An.transpose(0, 2, 1)
            tri = np.triu_indices(K, 1)
            cv = cos[:, tri[0], tri[1]]
            acc[v]["pairwise_cos_mean"].extend(cv.mean(axis=1).tolist())
            acc[v]["frac_cos_gt_0.9"].extend((cv > 0.9).mean(axis=1).tolist())
    out = {}
    for v in variants:
        out[v] = {m: _stats(acc[v][m]) for m in metrics}
    out["note"] = ("每图把 z_s[i] (32×1248) 看作 32 个样本、1248 维；奇异值平方就是跨位置"
                   "能量的谱，最大秩 32。raw=不处理；minus_posmean=减去数据集的逐位置均值"
                   "（任务描述口径）；minus_imgmean=减去该图 32 个 token 的均值。")
    return out


# ----------------------------------------------------------------------------
# split-half 抽样稳定性探针（只跑若干代表位置，避免全量 32 位置重复特征分解）
# ----------------------------------------------------------------------------
def split_half_probe(tokens_path, positions, out_path, seed=0):
    z_s, z_cls, labels, paths, keys = load_tokens(tokens_path)
    rng = np.random.default_rng(seed)
    res = {"tokens_path": os.path.abspath(tokens_path),
           "positions": list(positions), "seed": seed,
           "note": ("把该位置的 5000 张随机对半（2500/2500）各拟合一次 PCA：报告 rank@阈值、"
                    "两半 top-224 子空间主角度、前 100 个解释方差比的 L1 相对差。"
                    "注意 n 减半本身会让 rank@固定阈值系统性变小（有限样本效应），"
                    "两半之间的一致性反映的是抽样稳定性，不能直接与 5000 样本的 rank 比较。"),
           "results": {}}
    for k in positions:
        Z = z_s[:, k, :].astype(np.float64)
        r = analyze_matrix(Z, split_half=True, rng=rng, top_r_basis=224)
        r.pop("_basis", None)
        _print(f"[probe pos {k}] full rank@90/95/99/99.9="
               + "/".join(str(r["rank_at"][f"{t:.3f}"]) for t in THRESHOLDS)
               + f" | halves rank@95="
               + f"{r['split_half']['fit_a']['rank_at']['0.950']}/"
               + f"{r['split_half']['fit_b']['rank_at']['0.950']}"
               + f" | angle(top224)={r['split_half']['mean_angle_deg_top224']:.2f}deg")
        res["results"][str(k)] = r
    summary = {
        "abs_rank95_diff": _stats([res["results"][str(k)]["split_half"]["abs_rank_diff"]["0.950"]
                                   for k in positions]),
        "mean_angle_deg_top224": _stats([res["results"][str(k)]["split_half"]["mean_angle_deg_top224"]
                                         for k in positions]),
        "top100_evr_l1_rel_diff": _stats([res["results"][str(k)]["split_half"]["top100_evr_l1_rel_diff"]
                                          for k in positions]),
    }
    res["summary"] = summary
    with open(out_path, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    _print(f"[probe] split-half stability -> {out_path}")
    _print("[probe] summary: " + json.dumps(
        {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in summary.items()},
        ensure_ascii=False))
    return res


# ----------------------------------------------------------------------------
# 存盘
# ----------------------------------------------------------------------------
def verify_saved(out_dir, z_s, res):
    """复核落盘 npz：主成分正交性、前 64 主成分复现的方差保留率、特征值一致性。"""
    path = os.path.join(out_dir, "pca_perpos_components.npz")
    with np.load(path) as d:
        top = np.asarray(d["top64_components"], dtype=np.float64)   # (K,64,D)
        means = np.asarray(d["pos_means"], dtype=np.float64)        # (K,D)
        eig = np.asarray(d["eigenvalues_centered"], dtype=np.float64)
    K, nc, D = top.shape
    eye = np.eye(nc)
    orth_err = float(max(np.abs(top[k] @ top[k].T - eye).max() for k in range(K)))
    # 位置 0/15/31：用主成分投影后的方差和 vs spectra 的 top64 保留率
    rel_errs, sample_errs = [], []
    for k in (0, min(15, K - 1), K - 1):
        Z = np.asarray(z_s[:, k, :], dtype=np.float64)
        Zc = Z - Z.mean(axis=0)
        proj = Zc @ top[k].T
        var_proj = float(proj.var(axis=0, ddof=1).sum())
        var_tot = float(Zc.var(axis=0, ddof=1).sum())
        got = var_proj / var_tot
        want = res["positions"][k]["top_var"][str(nc)]
        rel_errs.append(abs(got - want) / max(want, 1e-300))
        sample_errs.append(abs(float(eig[k, 0]) -
                               res["positions"][k]["eigenvalues_centered_sample"]["1"])
                           / max(abs(float(eig[k, 0])), 1e-300))
    return {
        "components_npz_orthonormality_max_err": orth_err,
        "components_npz_top64_recon_max_rel_err": float(max(rel_errs)),
        "eig_npz_vs_json_sample_max_rel_err": float(max(sample_errs)),
        "components_npz_shape": [int(K), int(nc), int(D)],
    }


def save_outputs(res, out_dir, meta, z_s=None):
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, "pca_perpos_components.npz"),
        pos_means=res["pos_means"].astype(np.float32),
        top64_components=res["pos_top"].astype(np.float32),
        eigenvalues_centered=res["eig_full"],
        explained_var_ratio_centered=(res["eig_full"] /
                                      res["eig_full"].sum(axis=1, keepdims=True)),
    )
    if z_s is not None:
        res["checks"].update(verify_saved(out_dir, z_s, res))
    meta = dict(meta)
    meta["checks"] = res["checks"]
    spectra = {
        "meta": meta,
        "checks": res["checks"],
        "reference_numbers": REFERENCE,
        "positions": res["positions"],
        "aggregate": res["aggregate"],
    }
    with open(os.path.join(out_dir, "pca_perpos_spectra.json"), "w") as f:
        json.dump(spectra, f, ensure_ascii=False, indent=1)
    crosspos = {"meta": meta, "checks": res["checks"], **res["crosspos"]}
    if res["z_cls"]:
        crosspos["z_cls"] = res["z_cls"]
    with open(os.path.join(out_dir, "crosspos.json"), "w") as f:
        json.dump(crosspos, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
        json.dump({"meta": meta, "checks": res["checks"],
                   "elapsed_compute_s": res["elapsed_compute_s"]},
                  f, ensure_ascii=False, indent=1)
    _print(f"[save] -> {out_dir}/pca_perpos_spectra.json, pca_perpos_components.npz, "
           f"crosspos.json, run_meta.json")


# ----------------------------------------------------------------------------
# 作图
# ----------------------------------------------------------------------------
def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "legend.fontsize": 6.8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 120, "savefig.dpi": 300, "pdf.fonttype": 42,
        "ps.fonttype": 42, "axes.unicode_minus": False,
    })
    return plt


def _savefig(fig, fig_dir, name):
    os.makedirs(fig_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"{name}.{ext}"), bbox_inches="tight")
    _print(f"[fig] {os.path.join(fig_dir, name)}.png/.pdf")


def _spectra_arrays(spectra):
    """从 json 抽样点还原谱曲线（evr / cum 都是相对真实总方差，勿重新归一化抽样值）。"""
    pos = spectra["positions"]
    D = spectra["checks"]["dim"]
    ranks = np.array(sorted(int(k) for k in pos[0]["evr_sample"].keys()))
    evr = np.array([[p["evr_sample"][str(r)] for r in ranks] for p in pos])
    cum = np.array([[p["cum_var_sample"][str(r)] for r in ranks] for p in pos])
    return ranks, evr, cum, D


def fig_pca_spectrum(spectra, fig_dir):
    plt = _setup_mpl()
    ranks, evr, cum, D = _spectra_arrays(spectra)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True)
    for k in range(evr.shape[0]):
        ax[0].semilogy(ranks, evr[k], color="0.75", lw=0.5, alpha=0.7)
    med = np.median(evr, axis=0)
    ax[0].semilogy(ranks, med, color="C3", lw=1.6, label="median over 32 positions")
    ax[0].axvline(224, color="C0", ls="--", lw=0.8)
    ax[0].axvline(448, color="C0", ls=":", lw=0.9)
    ax[0].text(232, med.max() * 0.35, "224", color="C0", fontsize=6.5)
    ax[0].text(456, med.max() * 0.12, "448", color="C0", fontsize=6.5)
    ax[0].set_xlabel("principal-component index $i$")
    ax[0].set_ylabel("explained variance ratio $\\lambda_i/\\sum_j\\lambda_j$")
    ax[0].set_xlim(0, D)
    ax[0].legend(loc="upper right", frameon=False)
    ax[0].set_title("(a) per-position eigenvalue spectrum (centered PCA)", loc="left")

    for k in range(cum.shape[0]):
        ax[1].plot(ranks, cum[k], color="0.75", lw=0.5, alpha=0.7)
    ax[1].plot(ranks, np.median(cum, axis=0), color="C3", lw=1.6)
    for t, c in ((0.90, "C0"), (0.95, "C1"), (0.99, "C2")):
        ax[1].axhline(t, color=c, ls="--", lw=0.7)
        ax[1].text(8, t + 0.006, f"{t:.0%}", color=c, fontsize=6.5)
    ax[1].axvline(224, color="C0", ls="--", lw=0.8)
    ax[1].axvline(448, color="C0", ls=":", lw=0.9)
    ax[1].set_xlabel("number of retained principal components $r$")
    ax[1].set_ylabel("cumulative explained variance")
    ax[1].set_xlim(0, D)
    ax[1].set_ylim(max(0.0, np.median(cum[:, 0]).min() - 0.05, 0.3), 1.005)
    ax[1].set_title("(b) cumulative explained variance", loc="left")
    _savefig(fig, fig_dir, "fig_pca_spectrum")
    plt.close(fig)


def fig_rank_at_threshold(spectra, fig_dir):
    plt = _setup_mpl()
    pos = spectra["positions"]
    K = len(pos)
    D = spectra["checks"]["dim"]
    thrs = [f"{t:.3f}" for t in THRESHOLDS]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True)
    x = np.arange(K)
    for t, c, m in zip(thrs, ("C0", "C1", "C2", "C3"), ("o", "s", "^", "D")):
        y = [p["rank_at"][t] for p in pos]
        ax[0].plot(x, y, marker=m, ms=2.6, lw=1.0, color=c,
                   label=f"rank@{float(t):.1%}")
    ax[0].axhline(224, color="0.4", ls="--", lw=0.7)
    ax[0].axhline(448, color="0.4", ls=":", lw=0.8)
    ax[0].set_xlabel("token position $k$")
    ax[0].set_ylabel("components needed to reach threshold")
    ax[0].set_yscale("log")
    ax[0].set_xlim(-0.6, K - 0.4)
    ax[0].legend(loc="lower right", frameon=False, ncol=2, fontsize=6.3)
    ax[0].set_title("(a) rank vs token position", loc="left")
    # 右侧轴标出 zs224 / zs448 的每 token 维数（与左侧同 log 尺度）
    ax0r = ax[0].twinx()
    ax0r.set_yscale("log")
    ax0r.set_ylim(ax[0].get_ylim())
    ax0r.set_yticks([224.0, 448.0])
    ax0r.set_yticklabels(["224\n(zs224)", "448\n(zs448)"], fontsize=6.3)
    ax0r.minorticks_off()
    ax0r.tick_params(axis="y", length=2.5, pad=1)
    ax0r.spines["top"].set_visible(False)

    data = [[p["rank_at"][t] for p in pos] for t in thrs]
    bp = ax[1].boxplot(data, widths=0.5, showfliers=False, patch_artist=True,
                       medianprops=dict(color="C3", lw=1.0))
    for patch, c in zip(bp["boxes"], ("C0", "C1", "C2", "C3")):
        patch.set_facecolor(c)
        patch.set_alpha(0.45)
        patch.set_edgecolor(c)
    for j, t in enumerate(thrs):
        med = spectra["aggregate"]["rank_at"][t]["median"]
        ax[1].text(j + 1, med, f"{med:.0f}", ha="center", va="bottom", fontsize=6.5)
    ax[1].axhline(224, color="0.4", ls="--", lw=0.7)
    ax[1].axhline(448, color="0.4", ls=":", lw=0.8)
    ax[1].set_xticks(range(1, len(thrs) + 1))
    ax[1].set_xticklabels([f"rank@{float(t):.1%}" for t in thrs], fontsize=6.8)
    ax[1].set_ylabel("components ($D$ = 1248)")
    ax[1].set_yscale("log")
    ax[1].set_title("(b) distribution over 32 positions (median annotated)", loc="left")
    _savefig(fig, fig_dir, "fig_rank_at_threshold")
    plt.close(fig)


def fig_crosspos_angles(crosspos, fig_dir):
    plt = _setup_mpl()
    cp = crosspos
    K = np.array(cp["position_means"]["cos_matrix"]).shape[0]
    ang224 = np.array(cp["principal_angles"]["224"]["mean_angle_deg_matrix"])
    ov224 = np.array(cp["principal_angles"]["224"]["overlap_matrix"])
    cos_mu = np.array(cp["position_means"]["cos_matrix"])
    fig, ax = plt.subplots(1, 3, figsize=(10.2, 2.85), constrained_layout=True)
    im0 = ax[0].imshow(ang224, cmap="viridis", vmin=0, vmax=90)
    cb0 = fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.02)
    cb0.set_label("mean principal angle (deg)", fontsize=7)
    ax[0].set_title("(a) top-224 subspaces: mean principal angle", loc="left")

    im1 = ax[1].imshow(ov224, cmap="magma", vmin=0, vmax=1)
    cb1 = fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.02)
    cb1.set_label("subspace overlap $\\|U_i^\\top U_j\\|_F^2/r$", fontsize=7)
    ax[1].set_title("(b) top-224 subspace overlap", loc="left")

    im2 = ax[2].imshow(cos_mu, cmap="RdBu_r", vmin=-1, vmax=1)
    cb2 = fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.02)
    cb2.set_label("cosine between position means", fontsize=7)
    ax[2].set_title("(c) position-mean cosine", loc="left")
    ticks = list(range(0, K, 4))
    for a in ax:
        a.set_xticks(ticks)
        a.set_yticks(ticks)
        a.set_xlabel("token position $k$")
        a.grid(False)
    ax[0].set_ylabel("token position $k$")
    _savefig(fig, fig_dir, "fig_crosspos_angles")
    plt.close(fig)


def fig_var_at_448_224(spectra, crosspos, fig_dir):
    plt = _setup_mpl()
    pos = spectra["positions"]
    K = len(pos)
    agg = spectra["aggregate"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    x = np.arange(K)
    w = 0.27
    for j, (r, c, lab) in enumerate(zip(("224", "448", "624"),
                                        ("C0", "C1", "C2"),
                                        ("$r$=224", "$r$=448", "$r$=624"))):
        y = [p["top_var"][r] for p in pos]
        ax[0].bar(x + (j - 1) * w, y, width=w, color=c, alpha=0.75,
                  edgecolor="none", label=lab)
    lo_a = min(min(p["top_var"][r] for p in pos) for r in ("224", "448", "624"))
    for t, c in ((0.90, "0.3"), (0.95, "0.5"), (0.99, "0.7")):
        ax[0].axhline(t, color=c, ls="--", lw=0.7)
        ax[0].text(K - 0.6, t + 0.004, f"{t:.0%}", ha="right", fontsize=6.2, color="0.3")
    ax[0].set_xlabel("token position $k$")
    ax[0].set_ylabel("cumulative explained variance\n(per-position centered PCA)")
    ax[0].set_ylim(max(0.0, lo_a - 0.05), 1.02)
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3,
                 frameon=False)
    ax[0].set_title("(a) variance retained at $r$=224/448/624", loc="left")

    # 面板 b：同一 r，不同口径的 token 方差保留率（与 448/224 降维臂对照）
    pool = crosspos["pooled"]
    group_labels = ["$r$=224", "$r$=448"]
    series = [
        ("per-position PCA (median, p10-p90)", "C0",
         [agg["top_var"]["224"]["median"], agg["top_var"]["448"]["median"]],
         [agg["top_var"]["224"]["p10"], agg["top_var"]["448"]["p10"]],
         [agg["top_var"]["224"]["p90"], agg["top_var"]["448"]["p90"]]),
        ("pooled, per-position centered", "C1",
         [pool["perpos_centered"]["top_var"]["224"],
          pool["perpos_centered"]["top_var"]["448"]], None, None),
        ("pooled, global centered", "C2",
         [pool["global_centered"]["top_var"]["224"],
          pool["global_centered"]["top_var"]["448"]], None, None),
        ("pooled, raw (no centering)", "C3",
         [pool["raw"]["top_var"]["224"], pool["raw"]["top_var"]["448"]], None, None),
    ]
    gx = np.arange(len(group_labels))
    width = 0.19
    for j, (lab, c, y, lo, hi) in enumerate(series):
        if lo is not None:
            ax[1].bar(gx + (j - 1.5) * width, y, width=width, color=c, alpha=0.8,
                      edgecolor="none", label=lab,
                      yerr=[np.array(y) - np.array(lo), np.array(hi) - np.array(y)],
                      capsize=2, error_kw=dict(lw=0.7))
        else:
            ax[1].bar(gx + (j - 1.5) * width, y, width=width, color=c, alpha=0.8,
                      edgecolor="none", label=lab)
    all_y = []
    for lab, c, y, lo, hi in series:
        all_y.extend(y)
        if lo is not None:
            all_y.extend(lo)
    lo_b = max(0.0, min(all_y) - 0.06)
    for j, (lab, c, y, lo, hi) in enumerate(series):
        for xi, yi in zip(gx + (j - 1.5) * width, y):
            ax[1].text(xi, yi + 0.006, f"{yi:.3f}", ha="center", va="bottom",
                       fontsize=5.4, rotation=90, color="0.25",
                       clip_on=True)
    ax[1].set_xticks(gx)
    ax[1].set_xticklabels(group_labels)
    ax[1].set_ylabel("token variance retained")
    ax[1].set_ylim(lo_b, min(1.12, max(all_y) + 0.06))
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
                 frameon=False, fontsize=5.8)
    ax[1].set_title("(b) retention at same $r$: per-position vs pooled", loc="left")
    _savefig(fig, fig_dir, "fig_var_at_448_224")
    plt.close(fig)


def make_figures(out_dir, fig_dir):
    with open(os.path.join(out_dir, "pca_perpos_spectra.json")) as f:
        spectra = json.load(f)
    with open(os.path.join(out_dir, "crosspos.json")) as f:
        crosspos = json.load(f)
    fig_pca_spectrum(spectra, fig_dir)
    fig_rank_at_threshold(spectra, fig_dir)
    fig_crosspos_angles(crosspos, fig_dir)
    fig_var_at_448_224(spectra, crosspos, fig_dir)
    _print(f"[fig] 4 张图 (png+pdf) -> {fig_dir}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", default=DEFAULT_TOKENS)
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    ap.add_argument("--fig_dir", default=None,
                    help="默认 <repo>/doc/2026-10-30/artifacts/zpca")
    ap.add_argument("--synthetic", type=int, default=0,
                    help="数据未到位时生成 N 张合成 token 跑通全流程（不写 sha256）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top_components", type=int, default=64,
                    help="components npz 里每位置保存的主成分个数")
    ap.add_argument("--no_split_half", action="store_true",
                    help="跳过 split-half 稳定性（省约 1/3 时间）")
    ap.add_argument("--no_figs", action="store_true")
    ap.add_argument("--figs_only", action="store_true", help="只按已有 json 重画图")
    ap.add_argument("--split_half_probe", default="",
                    help="逗号分隔的位置列表，只对这些位置做 split-half 抽样稳定性探针"
                         "（如 0,7,19,26,31），结果写 out_dir/split_half_probe.json")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = args.fig_dir or os.path.join(repo, "doc", "2026-10-30", "artifacts", "zpca")
    os.makedirs(args.out_dir, exist_ok=True)

    if args.figs_only:
        make_figures(args.out_dir, fig_dir)
        return

    if args.split_half_probe:
        positions = [int(x) for x in args.split_half_probe.split(",") if x.strip()]
        split_half_probe(args.tokens, positions,
                         os.path.join(args.out_dir, "split_half_probe.json"),
                         seed=args.seed)
        return

    t0 = time.time()
    if args.synthetic:
        z_s, z_cls, labels, paths, keys = make_synthetic(args.synthetic, seed=args.seed)
        z_cls = _squeeze_cls(z_cls)
        tokens_path = None
        tokens_sha = None
        _print(f"[synthetic] N={args.synthetic} seed={args.seed}")
    else:
        if not os.path.isfile(args.tokens):
            raise SystemExit(f"[data] token 文件不存在：{args.tokens}\n"
                             f"        T1 (remote-runner) 尚未完成；可先用 --synthetic 5000 "
                             f"把流程跑通。")
        tokens_path = os.path.abspath(args.tokens)
        _print(f"[data] {tokens_path} sha256 ...")
        tokens_sha = _sha256(tokens_path)
        z_s, z_cls, labels, paths, keys = load_tokens(tokens_path)
        _print(f"[data] sha256={tokens_sha[:16]}... keys={keys}")

    meta = {
        "script": os.path.abspath(__file__),
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "tokens_path": tokens_path,
        "tokens_sha256": tokens_sha,
        "synthetic": bool(args.synthetic),
        "synthetic_seed": args.seed if args.synthetic else None,
        "shape": list(z_s.shape),
        "z_cls_shape": list(z_cls.shape) if z_cls is not None else None,
        "dtype_in": str(z_s.dtype),
        "n_images": int(z_s.shape[0]),
        "n_positions": int(z_s.shape[1]),
        "dim": int(z_s.shape[2]),
        "n_labels": int(len(labels)) if labels is not None else None,
        "n_paths": int(len(paths)) if paths is not None else None,
        "npz_keys": keys,
        "numpy_version": np.__version__,
        "python": os.sys.version.split()[0],
        "thresholds": list(THRESHOLDS),
        "top_ks": list(TOP_KS),
        "top_components_saved": args.top_components,
        "split_half": not args.no_split_half,
        "compute_policy": {
            "center": "逐位置减去该位置在 5000 张 val 上的均值；协方差 /(N-1)",
            "precision": "输入 fp16 -> 逐位置 astype(float64) 后做 Gram / eigh",
            "pca_basis": "在测试集（IN-100 val 5000）上拟合 —— oracle/transductive 上界",
            "fp16_boundary": "fp16 相对步长 ~4.9e-4，影响最低特征值/rank@99.9 尾端，"
                             "对 rank@90/95/99 与 224/448 覆盖影响可忽略（另见 split-half）",
        },
    }

    res = main_analysis(z_s, z_cls, args.out_dir,
                        split_half=not args.no_split_half,
                        seed=args.seed,
                        top_components=args.top_components)
    meta["elapsed_total_s"] = time.time() - t0
    save_outputs(res, args.out_dir, meta, z_s=z_s)

    # 数字自洽快速复核（写入 stdout）
    chk = res["checks"]
    _print(f"[check] cum_monotone={chk['cum_monotone']} "
           f"eigsum_vs_trace_max_rel_err={chk['eigsum_vs_trace_max_rel_err']:.3e} "
           f"evr_sum_max_abs_dev={chk['evr_sum_max_abs_dev']:.3e}")
    agg = res["aggregate"]
    _print("[summary] rank@90/95/99/99.9 median = "
           + "/".join(f"{agg['rank_at'][f'{t:.3f}']['median']:.0f}" for t in THRESHOLDS)
           + f"; PR median={agg['participation_ratio']['median']:.1f}"
           + f"; top224 median={agg['top_var']['224']['median']:.4f}"
           + f"; top448 median={agg['top_var']['448']['median']:.4f}")
    _print(f"[done] {meta['elapsed_total_s']:.1f}s -> {args.out_dir}")

    if not args.no_figs:
        make_figures(args.out_dir, fig_dir)


if __name__ == "__main__":
    main()
