#!/usr/bin/env python3
"""qwen_pca_tokens.py — 在 Qwen3.8-27B 视觉 token（IN-100 训练集，全局池化）上做 PCA。

口径（2026-10-30 用户）: 同图 49 个 token 按顺序拼接；**整个训练集**的所有 token
一起（不分位置、全局池化）算 PCA；给出 1σ/2σ/3σ/全部 对应的主成分数。

「σ」的权威定义（报告必须写清）: 高斯累计质量口径 —— 保留累计解释方差比例
  1σ = 68.27%、2σ = 95.45%、3σ = 99.73%、全部 = 100%（r = token 维度 5120）。
备择口径（仅透明性，不用于实验）:
  · alt_mean_mult: λ_i ≥ k·mean(λ) 的个数（k=1/2/3，未标准化）；
  · alt_kaiser: 逐维标准化后的相关矩阵 eigenvalue ≥ 1/2/3 的个数。

数值实现: token 反量化后在 GPU 上用 float32 做块内 X^T X（块 ~4096 图），块间累加
到 float64；**中心化技巧**：第一块估出参考中心 c，后续块先减去 c 再乘，避免
mean^2 与方差之间的 float32 相消（最终 cov = A/n - (μ-c)(μ-c)^T 精确还原）。
特征分解用 torch.linalg.eigh 的 **float64**（GPU，5120 维 ~1.2 s），因此
eigvals/components 与 numpy float64 独立复算的差 ~1e-13。

产物（--out_dir）:
  mean.npy                  (5120,) float32   池化均值
  components.npy            (5120,5120) float32  列 = 降序特征向量（PC1 在第一列）
  eigvals.npy               (5120,) float64   降序特征值
  cumvar.npy                (5120,) float64   累计解释方差
  pca_stats.json            n_tokens / 阈值 r / PR / entropy rank / top-k share /
                            alt_mean_mult / alt_kaiser / trunc_rmse / token_meta
  arms.json                 1σ/2σ/3σ/all 的 r、cumvar_at、eigval_at、定义（探针组直接读）
  quant_fidelity.json       int8 vs fp16 子集：量化误差 / 子集谱阈值一致性 /
                            f32 累积协方差 vs f64 独立复算的相对误差
  per_position_eigvals.npz  (49,5120) float32（--per_position）
  per_position_means.npy    (49,5120) float32 每位置均值（--per_position）
  per_position_stats.json   逐位置阈值 r / 方差占比 / 位置常量占比（混合方差分解）/
                            PC1..PC4 与位置均值子空间的重叠（--per_position）
  pca_stats_poscentered.json / poscentered_eigvals.npy / poscentered_cumvar.npy
                            位置中心化池化 PCA（secondary：先减 μ_p 再池化；
                            等价于 mean_p cov_p，用于分离位置/偏置结构与图像内容）

自测（CPU，不占 GPU）: 合成数据 --dim 64 --device cpu --chunk 8，独立 numpy 复算校验
（真实规模 5120 维用 GPU float64 eigh，已单独验证过精度）。

用法:
  python tools/qwen_pca_tokens.py --out_dir /root/autodl-tmp/qwen_vtc/pca --per_position
  # 自测（合成小数据）:
  python tools/qwen_pca_tokens.py --prep .../pca/_unittest/tokens \
      --out_dir .../pca/_unittest/out --per_position
"""
import argparse
import json
import os
import time

import numpy as np
import torch

R = "/root/autodl-tmp/qwen_vtc"
N_TOK = 49
MEM_DIM = 5120
SIGMA_TARGETS = {"1sigma": 0.6827, "2sigma": 0.9545, "3sigma": 0.9973}
REF_TARGETS = {"90": 0.90, "95": 0.95, "99": 0.99, "999": 0.999}
ALL_TARGETS = {**SIGMA_TARGETS, **REF_TARGETS}
TOPK_SHARES = (1, 4, 16, 49, 64, 128, 256, 448, 512, 1024, 2048)
SIGMA_NOTE = ("高斯累计质量口径: 1σ/2σ/3σ = 保留累计解释方差 68.27%/95.45%/99.73% "
              "对应的主成分数 r; all = 5120（100%）")


def set_precision(device):
    """关掉 TF32，保证 float32 块内 matmul 精度。"""
    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def load_meta(prep):
    p = os.path.join(prep, "meta.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:                                   # noqa: BLE001
        return {"_meta_read_error": str(e)}


def load_quant_refs(prep, device):
    """B2 量化（2026-10-30 修订版）的 refs: x = (q*sc)*sd + mu_p。
    quant_ref.npz 含 mu_p (49,D) f32、sd (D,) f32；不存在则退回 v1 per-token absmax（x=q*sc）。"""
    p = os.path.join(prep, "quant_ref.npz")
    if not os.path.exists(p):
        return None, {"scheme": "v1_per_token_absmax_int8"}
    z = np.load(p)
    mu = torch.from_numpy(z["mu_p"].astype(np.float32)).to(device)
    sd = torch.from_numpy(z["sd"].astype(np.float32)).to(device)
    info = {"scheme": "b2_posmean_perchannel_pertoken_int8",
            "mu_p_shape": list(z["mu_p"].shape), "sd_shape": list(z["sd"].shape),
            "mu_p_absmax": float(np.abs(z["mu_p"]).max()),
            "sd_median": float(np.median(z["sd"])), "sd_max": float(z["sd"].max())}
    return (mu, sd), info


def dequant_gpu(q, s, device, refs=None):
    """q: (b,49,D) int8 numpy; s: (b,49) float numpy → (b*49,D) float32 on device.
    refs=(mu_p (49,D) on device, sd (D,) on device) → B2: x = (q*sc)*sd + mu_p。"""
    qg = torch.from_numpy(np.ascontiguousarray(q)).to(device, non_blocking=True)
    sg = torch.from_numpy(np.ascontiguousarray(s, dtype=np.float32)) \
        .to(device, non_blocking=True).reshape(-1, 1)
    x = qg.reshape(-1, MEM_DIM).to(torch.float32)
    x.mul_(sg)
    if refs is not None:
        mu, sd = refs
        x.mul_(sd)
        x.view(-1, N_TOK, MEM_DIM).add_(mu)          # mu_p 按位置广播（x 连续，view 安全）
    del qg
    return x


def cov_from_accum(A_gpu, n, shift):
    """A_gpu = Σ (x-c)(x-c)^T (float32, GPU); shift = (μ-c) 长度 D 的 float64。
    → cov = A/n - shift·shift^T（float64 numpy，对称化）。"""
    A = A_gpu.double().cpu().numpy()
    cov = A / float(n)
    cov -= np.outer(shift, shift)
    return 0.5 * (cov + cov.T)


def eigh_desc(cov_np, device, vectors=True):
    """cov_np float64 numpy → (降序特征值 float64[5120], 特征向量 float32 (D,D) or None)。"""
    c = torch.from_numpy(np.ascontiguousarray(cov_np, dtype=np.float64)).to(device)
    if vectors:
        ev, V = torch.linalg.eigh(c)
        V = V.flip(1).contiguous()
        Vf = V.to(torch.float32).cpu().numpy().copy()
        del V
    else:
        ev = torch.linalg.eigvalsh(c)
        Vf = None
    ev = ev.flip(0).clamp_min(0.0).cpu().numpy().astype(np.float64)
    del c
    return ev, Vf


def dims_at(cum, targets):
    out = {}
    for name, t in targets.items():
        r = int(np.searchsorted(cum, t) + 1)
        r = min(r, len(cum))
        out[name] = {"target": float(t), "r": int(r), "cumvar_at_r": float(cum[r - 1])}
    return out


def write_stats(out_dir, st):
    with open(os.path.join(out_dir, "pca_stats.json"), "w") as f:
        json.dump(st, f, indent=1, ensure_ascii=False)


def write_arms(out_dir, st):
    """arms.json —— 探针组的唯一 r 来源。"""
    dt = st["dims_at_thresholds"]
    arms = {
        "1sigma": int(dt["1sigma"]["r"]),
        "2sigma": int(dt["2sigma"]["r"]),
        "3sigma": int(dt["3sigma"]["r"]),
        "all": int(MEM_DIM),
        "cumvar_at": {k: float(dt[k]["cumvar_at_r"]) for k in ("1sigma", "2sigma", "3sigma")},
        "cumvar_targets": {k: float(SIGMA_TARGETS[k]) for k in SIGMA_TARGETS},
        "eigval_at": {k: float(st["eigvals_at_r"].get(k, 0.0)) for k in ("1sigma", "2sigma", "3sigma")},
        "argmax_ratio_eig_at_r": {k: float(st["argmax_ratio_eig_at_r"].get(k, 0.0))
                                  for k in ("1sigma", "2sigma", "3sigma")},
        "ref_thresholds_r": {k: int(dt[k]["r"]) for k in REF_TARGETS},
        "alt_mean_mult": st["alt_mean_mult"],
        "alt_kaiser": st["alt_kaiser"],
        "definition": "cumulative explained variance 0.6827/0.9545/0.9973",
        "definition_long": SIGMA_NOTE,
        "sampling": st["sampling"],
        "n_tokens": st["n_tokens"],
        "n_train_images": st["n_train_images"],
        "dim": MEM_DIM,
        "source": "qwen_pca_tokens.py",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(out_dir, "arms.json"), "w") as f:
        json.dump(arms, f, indent=1, ensure_ascii=False)
    print("[pca] arms:", json.dumps({k: arms[k] for k in ("1sigma", "2sigma", "3sigma", "all",
                                                          "cumvar_at")}, ensure_ascii=False),
          flush=True)
    return arms


# ───────────────────────── 量化保真度 ─────────────────────────
def quant_fidelity(args, out_dir, tok, scale, N_full, cum_full, c_glob, c_glob_gpu, device,
                   refs=None):
    sub_path = os.path.join(args.prep, "train_fp16_subset.npy")
    if not os.path.exists(sub_path):
        print("[pca] no train_fp16_subset.npy → skip quant_fidelity", flush=True)
        return None
    sub = np.load(sub_path)                                   # (S,49,D) float16
    idx_path = os.path.join(args.prep, "train_fp16_subset_idx.npy")
    if os.path.exists(idx_path):
        idx = np.load(idx_path).astype(np.int64)
    else:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(N_full, size=len(sub), replace=False))
    S = len(sub)
    b_fid = max(1, min(int(args.fid_batch), S))
    D = MEM_DIM
    max_abs, sse, ref = 0.0, 0.0, 0.0
    A_sub = None                                               # f32 Σ(x-c)(x-c)^T (fp16 子集)
    A_int = None                                               # f32 Σ(x-c)(x-c)^T (同一子集的 int8 反量化)
    s_sub = torch.zeros(D, dtype=torch.float64)
    s_int = torch.zeros(D, dtype=torch.float64)
    n_sub = 0
    n_chunks = (S + b_fid - 1) // b_fid
    for j in range(0, S, b_fid):
        hi = min(j + b_fid, S)
        q = np.asarray(tok[idx[j:hi]], dtype=np.int8)
        s = np.asarray(scale[idx[j:hi]], dtype=np.float32)
        x = dequant_gpu(q, s, device, refs)                    # (b*49,D) f32
        subc = torch.from_numpy(np.ascontiguousarray(sub[j:hi], dtype=np.float32)) \
            .to(device).reshape(-1, D)
        d = x - subc
        max_abs = max(max_abs, float(d.abs().max().item()))
        sse += float((d.double() ** 2).sum().item())
        ref += float((subc.double() ** 2).sum().item())
        del d
        s_sub += subc.sum(0, dtype=torch.float64).cpu()
        s_int += x.sum(0, dtype=torch.float64).cpu()
        xc = subc - c_glob_gpu
        g = xc.t() @ xc
        A_sub = g if A_sub is None else A_sub.add_(g)
        xi = x - c_glob_gpu
        g2 = xi.t() @ xi
        A_int = g2 if A_int is None else A_int.add_(g2)
        n_sub += xc.shape[0]
        del x, subc, xc, g, xi, g2
        print(f"[pca] fidelity chunk {j//b_fid+1}/{n_chunks}", flush=True)
    sub_mean = (s_sub / n_sub).numpy()                         # float64
    # 我们的 f32 累积路径
    cov_sub = cov_from_accum(A_sub, n_sub, sub_mean - c_glob)
    del A_sub
    # numpy float64 独立复算子集协方差（检验 f32 累积精度）
    A_ref = np.zeros((D, D), dtype=np.float64)
    s_ref = np.zeros(D, dtype=np.float64)
    for j in range(0, S, b_fid):
        hi = min(j + b_fid, S)
        X = np.asarray(sub[j:hi], dtype=np.float64).reshape(-1, D)
        s_ref += X.sum(0)
        Xc = X - c_glob
        A_ref += Xc.T @ Xc
        del X, Xc
    mean_ref = s_ref / n_sub
    cov_ref = A_ref / n_sub - np.outer(mean_ref - c_glob, mean_ref - c_glob)
    cov_ref = 0.5 * (cov_ref + cov_ref.T)
    del A_ref
    rel_cov = float(np.linalg.norm(cov_sub - cov_ref) / np.linalg.norm(cov_ref))
    # 同一子集、同一采样下的 int8 反量化协方差 → 干净地隔离「量化噪声」影响
    int_mean = (s_int / n_sub).numpy()
    cov_int = cov_from_accum(A_int, n_sub, int_mean - c_glob)
    del A_int
    ev_ref, _ = eigh_desc(cov_ref, device, vectors=False)
    cum_ref = np.cumsum(ev_ref) / ev_ref.sum()
    ev_chk, _ = eigh_desc(cov_sub, device, vectors=False)
    cum_chk = np.cumsum(ev_chk) / ev_chk.sum()
    ev_int, _ = eigh_desc(cov_int, device, vectors=False)
    cum_int = np.cumsum(ev_int) / ev_int.sum()
    fid = {
        "n_imgs": int(S), "n_tokens": int(n_sub),
        "max_abs_diff": max_abs,
        "rel_rmse": float(np.sqrt(sse / max(ref, 1e-30))),
        "token_psnr_db": float(-20 * np.log10(max(np.sqrt(sse / max(ref, 1e-30)), 1e-12))),
        "subset_fp16_source": sub_path,
        "quant_scheme": (refs and "b2_posmean_perchannel_pertoken_int8") or "v1_per_token_absmax_int8",
        "subset_idx_head": [int(v) for v in idx[:10]],
        "pooled_cov_relerr_f32_pipeline_vs_f64": rel_cov,
        "variance_ratio_int8_over_fp16_same_subset": float(np.trace(cov_int) / np.trace(cov_ref)),
        "quant_noise_variance_fraction": float(max(np.trace(cov_int) - np.trace(cov_ref), 0.0)
                                               / np.trace(cov_int)),
        "note": ("int8 = 全量训练集的 per-token absmax 量化; fp16 子集 = 固定随机子集原始精度。"
                 "r_subset_fp16 vs r_subset_int8（同一批图、同一样本数）隔离量化噪声对阈值 r 的影响；"
                 "r_full_train_int8 用了全部训练图，与子集的差主要来自抽样噪声。"
                 "token 值重尾（absmax/rms~60），per-token absmax int8 对尾部阈值影响见 quant_noise_variance_fraction。"),
        "dims_at_thresholds": {},
    }
    for name, t in ALL_TARGETS.items():
        r_sub = int(np.searchsorted(cum_ref, t) + 1)
        r_chk = int(np.searchsorted(cum_chk, t) + 1)
        r_int = int(np.searchsorted(cum_int, t) + 1)
        r_full = int(np.searchsorted(cum_full, t) + 1)
        fid["dims_at_thresholds"][name] = {
            "target": float(t),
            "r_subset_fp16": r_sub,
            "r_subset_int8_same_images": r_int,
            "diff_quant_int8_minus_fp16": int(r_int - r_sub),
            "r_subset_fp16_f32_pipeline": r_chk,
            "r_full_train_int8": r_full,
            "diff_subset_minus_full": int(r_sub - r_full),
        }
    with open(os.path.join(out_dir, "quant_fidelity.json"), "w") as f:
        json.dump(fid, f, indent=1, ensure_ascii=False)
    print("[pca] fidelity:", json.dumps({k: fid[k] for k in
                                        ("n_imgs", "max_abs_diff", "rel_rmse", "token_psnr_db",
                                         "pooled_cov_relerr_f32_pipeline_vs_f64")},
                                       ensure_ascii=False), flush=True)
    return fid


# ───────────────────────── 逐位置 PCA（次要对照）─────────────────────────
def per_position_analysis(args, out_dir, tok, scale, N, mean_global, ev_global, c_pp,
                          pp_A, pp_sum, n_img, device):
    D = MEM_DIM
    t0 = time.time()
    ev_pp = np.zeros((N_TOK, D), dtype=np.float64)
    dims_pp, share_within, top1_ev = {}, {}, {}
    means_pp = np.zeros((N_TOK, D), dtype=np.float32)
    cov_traces = np.zeros(N_TOK, dtype=np.float64)
    C_pc = np.zeros((D, D), dtype=np.float64)                  # mean_p cov_p（位置中心化池化协方差）
    for p in range(N_TOK):
        mu_p = (pp_sum[p] / n_img).numpy()                     # float64
        means_pp[p] = mu_p.astype(np.float32)
        covp = cov_from_accum(pp_A[p], n_img, mu_p - c_pp[p])
        cov_traces[p] = float(np.trace(covp))
        C_pc += covp
        evp, _ = eigh_desc(covp, device, vectors=False)
        ev_pp[p] = evp
        cum_p = np.cumsum(evp) / evp.sum()
        dims_pp[str(p)] = {name: int(np.searchsorted(cum_p, t) + 1) for name, t in ALL_TARGETS.items()}
        dims_pp[str(p)]["cumvar_at_2sigma"] = float(cum_p[dims_pp[str(p)]["2sigma"] - 1])
        top1_ev[str(p)] = float(evp[0])
        del covp
        if (p + 1) % 7 == 0 or p == N_TOK - 1:
            print(f"[pca] per-position {p+1}/{N_TOK}  {(time.time()-t0)/60:.1f} min", flush=True)
    np.savez_compressed(os.path.join(out_dir, "per_position_eigvals.npz"),
                        eigvals=ev_pp.astype(np.float32))
    np.save(os.path.join(out_dir, "per_position_means.npy"), means_pp)
    # 位置方差占比（位置内，Σ=1）
    share_within = cov_traces / cov_traces.sum()

    # ── 位置常量 vs 位置内内容：混合方差分解 ──
    # total = E_p[Var(x|p)] + Var(E[x|p]), 用 fp64
    mu_all = means_pp.astype(np.float64)                       # (49,D)
    dev = mu_all - mean_global[None, :]
    between = float((dev ** 2).sum() / N_TOK)                  # trace Var(E[x|p])
    within = float(cov_traces.mean())
    total = float(ev_global.sum())
    const_share = {
        "position_constant_between": between,
        "position_residual_within": within,
        "total_variance": total,
        "position_constant_frac": between / total,
        "position_residual_frac": within / total,
        "identity_residual_frac": (total - between - within) / total,
        "note": ("位置常量占比 = 位置均值之间的方差(混合方差分解的组间项) / 池化总方差；"
                 "位置内残差 = 平均逐位置协方差 trace。两者之和应等于 total（残差列见 identity_residual_frac）。"),
    }

    # ── 池化 PCA 前几个主成分 vs 位置均值子空间 ──
    M = mu_all - mean_global[None, :]                          # (49,D)
    U, S, Vh = np.linalg.svd(M, full_matrices=False)           # Vh: (49,D)
    comps = np.load(os.path.join(out_dir, "components.npy"), mmap_mode="r")
    Pcs = np.asarray(comps[:, :4], dtype=np.float64)           # (D,4)
    Q, _ = np.linalg.qr(Pcs)                                   # 正交基
    top4 = Vh[:4].T                                            # (D,4)
    Q4, _ = np.linalg.qr(top4)
    cross = Q.T @ Q4                                           # (4,4)
    sv_cross = np.linalg.svd(cross, compute_uv=False)
    ang = np.degrees(np.arccos(np.clip(sv_cross, -1, 1)))
    pc_top4_energy = float((np.linalg.norm(Q.T @ Q4) ** 2) / 4.0)   # PC1..4 被位置均值 top4 捕获的比例
    pos_top4_energy = float((np.linalg.norm(Q4.T @ Q) ** 2) / 4.0)  # 位置均值 top4 被 PC1..4 捕获的比例
    pc_vs_pos = {
        "pc_max_abs_cos_with_position_mean_singular_vectors": [float(v) for v in
                                                               np.abs(Q.T @ Q4).max(1)],
        "principal_angle_cos": [float(v) for v in sv_cross],
        "principal_angles_deg": [float(v) for v in ang],
        "position_mean_top4_captured_by_pc1_4": pos_top4_energy,
        "pc1_4_captured_by_position_mean_top4": pc_top4_energy,
        "position_mean_singular_values_top8": [float(v) for v in S[:8]],
        "note": ("位置均值矩阵 M(49x5120)=μ_p-μ̄ 的 SVD；比较 span(PC1..PC4) 与 M 的前 4 个右奇异向量。"
                 "值接近 1 说明池化主成分主要在描述位置结构（位置常量），接近 0 说明在描述图像内容。"),
    }

    st_pp = {
        "n_positions": N_TOK, "n_images": int(n_img), "dim": D,
        "grid": "7x7 merged tokens, row-major（与抽取脚本 model(pv,thw) 的输出顺序一致）",
        "dims_at_thresholds": dims_pp,
        "variance_share_within_sum1": {str(p): float(share_within[p]) for p in range(N_TOK)},
        "top1_eigval": top1_ev,
        "cov_trace": {str(p): float(cov_traces[p]) for p in range(N_TOK)},
        "position_constant_share": const_share,
        "pc_vs_position_mean": pc_vs_pos,
        "aggregate_dims": {name: {"min": int(min(dims_pp[str(p)][name] for p in range(N_TOK))),
                                  "median": float(np.median([dims_pp[str(p)][name]
                                                             for p in range(N_TOK)])),
                                  "max": int(max(dims_pp[str(p)][name] for p in range(N_TOK)))}
                           for name in ALL_TARGETS},
        "note": "49 个合并网格位置各自 5120 维 PCA；样本数 = 训练图数。仅特征值（阈值 r），不存逐位置特征向量。",
    }
    with open(os.path.join(out_dir, "per_position_stats.json"), "w") as f:
        json.dump(st_pp, f, indent=1, ensure_ascii=False)

    # ── 位置中心化池化 PCA（secondary，Lead 2026-10-30 追加）──
    # 精确恒等：x' = x - μ_p 后池化 ⇒ mean(x') = 0, cov(x') = mean_p cov_p = C_pc，
    # 所以无需二次读 token；直接由逐位置协方差合成（等价于再跑一遍池化 PCA）。
    C_pc /= N_TOK
    ev_pc, _ = eigh_desc(C_pc, device, vectors=False)
    cum_pc = np.cumsum(ev_pc) / ev_pc.sum()
    dims_pc = dims_at(cum_pc, ALL_TARGETS)
    np.save(os.path.join(out_dir, "poscentered_eigvals.npy"), ev_pc)
    np.save(os.path.join(out_dir, "poscentered_cumvar.npy"), cum_pc)
    ev_pc_sum = float(ev_pc.sum())
    pp_pc = ev_pc / ev_pc_sum
    cum_g = np.cumsum(ev_global) / ev_global.sum()
    dims_g = dims_at(cum_g, ALL_TARGETS)
    st_pc = {
        "sampling": "pooled_all_positions_after_per_position_mean_subtracted",
        "n_tokens": int(n_img * N_TOK), "n_images": int(n_img), "dim": D,
        "tokens_per_image": N_TOK,
        "description": ("先减每位置均值 μ_p（49×5120）再池化做 PCA；等价于对 mean_p cov_p 做 PCA。"
                        "与主口径（全局中心化池化）成对比较，用于分离位置/偏置结构。"),
        "total_variance": ev_pc_sum,
        "top1_share": float(ev_pc[0] / ev_pc_sum),
        "top_k_share": {str(k): float(ev_pc[:k].sum() / ev_pc_sum) for k in TOPK_SHARES if k <= len(ev_pc)},
        "participation_ratio": float(ev_pc_sum ** 2 / (ev_pc ** 2).sum()),
        "entropy_rank": float(np.exp(-(pp_pc * np.log(pp_pc + 1e-300)).sum())),
        "dims_at_thresholds": dims_pc,
        "cumvar_at": {name: float(cum_pc[dims_pc[name]["r"] - 1]) for name in ALL_TARGETS},
        "identity_check": {
            "trace_C_pc": float(np.trace(C_pc)),
            "mean_per_position_cov_trace": within,
            "rel_diff": float(abs(np.trace(C_pc) - within) / max(within, 1e-30)),
            "main_total_variance": float(ev_global.sum()),
            "poscentered_over_main_variance": ev_pc_sum / float(ev_global.sum()),
        },
        "vs_main": {name: {"main_r": int(dims_g[name]["r"]),
                           "poscentered_r": int(dims_pc[name]["r"]),
                           "ratio": (float(dims_pc[name]["r"]) / max(dims_g[name]["r"], 1))}
                    for name in ALL_TARGETS},
        "sigma_definition": SIGMA_NOTE,
        "note": "只加不减：主口径 mean.npy/components.npy/arms.json 不变；本文件仅供解释 r 的来源。",
    }
    with open(os.path.join(out_dir, "pca_stats_poscentered.json"), "w") as f:
        json.dump(st_pc, f, indent=1, ensure_ascii=False)
    print("[pca] poscentered dims:", json.dumps({name: dims_pc[name]["r"] for name in
                                                 ("1sigma", "2sigma", "3sigma")}, ensure_ascii=False),
          flush=True)
    print("[pca] poscentered vs main:", json.dumps(st_pc["vs_main"], ensure_ascii=False)[:500],
          flush=True)

    print("[pca] position_constant_share:", json.dumps(const_share, ensure_ascii=False),
          flush=True)
    print("[pca] pc_vs_position_mean:", json.dumps(pc_vs_pos, ensure_ascii=False)[:500], flush=True)
    return st_pp


# ───────────────────────── main ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(R, "pca"))
    ap.add_argument("--prep", default=os.path.join(R, "tokens"))
    ap.add_argument("--chunk", type=int, default=4096, help="每次读入的图片数（4096 图 ≈ 4.1GB f32 GPU）")
    ap.add_argument("--fid_batch", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0, help="只扫前 N 张图（自测/冒烟用）")
    ap.add_argument("--dim", type=int, default=0,
                    help="仅自测用：覆盖 token 维度（默认 5120；合成数据的最后一维必须等于它）")
    ap.add_argument("--per_position", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    t0 = time.time()
    global MEM_DIM
    if args.dim:
        MEM_DIM = int(args.dim)
    set_precision(args.device)
    dev = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    D = MEM_DIM

    tok_path = os.path.join(args.prep, "train_tokens_int8.npy")
    scale_path = os.path.join(args.prep, "train_scale.npy")
    tok = np.load(tok_path, mmap_mode="r")
    scale = np.load(scale_path, mmap_mode="r")
    n_full = tok.shape[0]
    assert tok.ndim == 3 and tuple(tok.shape[1:]) == (N_TOK, D), f"token shape {tok.shape}"
    assert tuple(scale.shape) == (n_full, N_TOK), f"scale shape {scale.shape}"
    N = min(args.limit, n_full) if args.limit else n_full
    meta = load_meta(args.prep)
    refs, quant_info = load_quant_refs(args.prep, dev)
    print(f"[pca] prep={args.prep} N={N}/{n_full} tokens={N*N_TOK} dim={D} "
          f"dtype={tok.dtype}/{scale.dtype} device={args.device} per_position={args.per_position} "
          f"quant={quant_info.get('scheme')}", flush=True)
    # 反量化口径自检：dequant vs fp16 子集前 3 张（口径错会立刻炸，避免白跑 10 min）
    sub_ck = os.path.join(args.prep, "train_fp16_subset.npy")
    idx_ck = os.path.join(args.prep, "train_fp16_subset_idx.npy")
    if os.path.exists(sub_ck) and os.path.exists(idx_ck):
        sub = np.load(sub_ck, mmap_mode="r")
        idx = np.load(idx_ck)[:3]
        x_ck = dequant_gpu(np.asarray(tok[idx], dtype=np.int8),
                           np.asarray(scale[idx], dtype=np.float32), dev, refs) \
            .cpu().numpy().reshape(len(idx), N_TOK, MEM_DIM)
        ref_ck = np.asarray(sub[:3], dtype=np.float32)
        rel_ck = float(np.linalg.norm(x_ck - ref_ck) / np.linalg.norm(ref_ck))
        max_ck = float(np.abs(x_ck - ref_ck).max())
        print(f"[pca] dequant self-check vs fp16 subset (3 imgs): rel_rmse={rel_ck:.5f} "
              f"max|d|={max_ck:.4f}", flush=True)
        if rel_ck > 0.10:
            raise RuntimeError(f"dequant 口径不自洽: rel_rmse={rel_ck:.4f}（检查 quant_ref.npz / scheme）")
    print(f"[pca] meta={json.dumps(meta, ensure_ascii=False)[:600]}", flush=True)

    s_acc = torch.zeros(D, dtype=torch.float64)                # Σx (float64)
    A_glob = None                                             # Σ(x-c)(x-c)^T (float32 GPU)
    pp_A = None
    pp_sum = None
    c_glob = None                                             # 参考中心 (float64 numpy)
    c_pp = None
    c_pp_gpu = None
    c_glob_gpu = None
    n_img = 0
    for lo in range(0, N, args.chunk):
        hi = min(lo + args.chunk, N)
        b = hi - lo
        q = np.asarray(tok[lo:hi], dtype=np.int8)              # (b,49,D) int8
        s = np.asarray(scale[lo:hi], dtype=np.float32)         # (b,49)
        x = dequant_gpu(q, s, dev, refs)                       # (b*49,D) float32
        del q
        x3 = x.reshape(b, N_TOK, D)
        s_acc += x.sum(0, dtype=torch.float64).cpu()
        if c_glob is None:
            c_glob = (x.sum(0, dtype=torch.float64).cpu().numpy() / x.shape[0])
            c_glob_gpu = torch.from_numpy(c_glob).to(dev).float()
            if args.per_position:
                c_pp = x3.sum(0, dtype=torch.float64).cpu().numpy() / b   # (49,D)
                c_pp_gpu = torch.from_numpy(c_pp).to(dev).float()
                pp_A = torch.zeros(N_TOK, D, D, dtype=torch.float32, device=dev)
                pp_sum = torch.zeros(N_TOK, D, dtype=torch.float64)
        if args.per_position:
            pp_sum.add_(x3.sum(0, dtype=torch.float64).cpu())
            for p in range(N_TOK):
                xpc = x3[:, p, :] - c_pp_gpu[p]
                pp_A[p].add_(xpc.t() @ xpc)
                del xpc
        x.sub_(c_glob_gpu)
        g = x.t() @ x
        A_glob = g if A_glob is None else A_glob.add_(g)
        del g, x3, x
        n_img += b
        el = time.time() - t0
        print(f"[pca] scanned {n_img}/{N} imgs ({n_img*N_TOK/1e6:.2f}M tokens) "
              f"{el/60:.1f} min  {n_img/max(el,1e-6):.0f} img/s", flush=True)
    n_tok = n_img * N_TOK
    mean = (s_acc / n_tok).numpy().astype(np.float64)
    np.save(os.path.join(args.out_dir, "mean.npy"), mean.astype(np.float32))
    cov = cov_from_accum(A_glob, n_tok, mean - c_glob)
    del A_glob
    print(f"[pca] pooled cov done, eigh (float64) ...", flush=True)
    ev, V = eigh_desc(cov, dev, vectors=True)
    cum = np.cumsum(ev) / ev.sum()
    np.save(os.path.join(args.out_dir, "eigvals.npy"), ev)
    np.save(os.path.join(args.out_dir, "components.npy"), V.astype(np.float32))
    np.save(os.path.join(args.out_dir, "cumvar.npy"), cum)
    del V

    dt = dims_at(cum, ALL_TARGETS)
    eigvals_at_r = {k: float(ev[v["r"] - 1]) for k, v in dt.items()}
    # 每个阈值 r 处的相邻特征值比 λ_r/λ_{r+1}（"肘部"程度）
    ratio = {}
    for k, v in dt.items():
        r = v["r"]
        ratio[k] = float(ev[r - 1] / max(ev[r] if r < len(ev) else 0.0, 1e-300))
    ev_sum = float(ev.sum())
    PR = float(ev_sum ** 2 / (ev ** 2).sum())
    p = ev / ev_sum
    ent_rank = float(np.exp(-(p * np.log(p + 1e-300)).sum()))
    st = {
        "n_tokens": int(n_tok), "n_train_images": int(n_img), "dim": D,
        "tokens_per_image": N_TOK, "sampling": "pooled_all_positions",
        "prep": args.prep, "script": "tools/qwen_pca_tokens.py",
        "elapsed_min": (time.time() - t0) / 60,
        "total_variance": ev_sum,
        "eigval_max": float(ev[0]), "eigval_min": float(ev[-1]),
        "top1_share": float(ev[0] / ev_sum),
        "top_k_share": {str(k): float(ev[:k].sum() / ev_sum) for k in TOPK_SHARES if k <= len(ev)},
        "participation_ratio": PR,
        "entropy_rank": ent_rank,
        "dims_at_thresholds": dt,
        "eigvals_at_r": eigvals_at_r,
        "argmax_ratio_eig_at_r": ratio,
        "alt_mean_mult": {str(k): int((ev >= k * ev.mean()).sum()) for k in (1, 2, 3)},
        "mean_eigval": float(ev.mean()),
        "eigval_decile": {str(i): float(ev[int(i / 10 * len(ev))]) for i in range(10)},
        "sigma_definition": SIGMA_NOTE,
        "token_meta": meta,
        "quant_scheme": quant_info,
    }
    # 截断重建 RMSE（token 空间逐元素）= sqrt(未解释方差 / D)
    st["trunc_rmse"] = {name: float(np.sqrt(max(ev_sum - cum[v["r"] - 1] * ev_sum, 0.0) / D))
                        for name, v in dt.items()}
    # Kaiser 口径（逐维标准化相关矩阵）
    sd = np.sqrt(np.clip(np.diag(cov).copy(), 1e-12, None))
    corr = cov / np.outer(sd, sd)
    ev_std, _ = eigh_desc(corr, dev, vectors=False)
    st["alt_kaiser"] = {str(k): int((ev_std >= k).sum()) for k in (1, 2, 3)}
    st["kaiser_note"] = "逐维标准化后 eigenvalue>=k 的个数（备择口径，非实验口径）"
    del corr
    write_stats(args.out_dir, st)
    arms = write_arms(args.out_dir, st)
    print("[pca] top_k_share:", json.dumps(st["top_k_share"], ensure_ascii=False), flush=True)
    print("[pca] PR=%.2f entropy_rank=%.2f top1_share=%.4f" % (PR, ent_rank, st["top1_share"]),
          flush=True)
    print("[pca] dims:", json.dumps({k: v["r"] for k, v in dt.items()}, ensure_ascii=False),
          flush=True)

    fid = quant_fidelity(args, args.out_dir, tok, scale, n_full, cum, c_glob, c_glob_gpu, dev,
                         refs)
    if fid is not None:
        st["quant_fidelity_brief"] = {
            "rel_rmse": fid["rel_rmse"], "max_abs_diff": fid["max_abs_diff"],
            "pooled_cov_relerr_f32_pipeline_vs_f64": fid["pooled_cov_relerr_f32_pipeline_vs_f64"],
            "quant_noise_variance_fraction": fid["quant_noise_variance_fraction"],
            "diff_r_at_sigma_quant": {k: fid["dims_at_thresholds"][k]["diff_quant_int8_minus_fp16"]
                                      for k in ("1sigma", "2sigma", "3sigma")},
            "diff_r_at_sigma_subset_vs_full": {k: fid["dims_at_thresholds"][k]["diff_subset_minus_full"]
                                               for k in ("1sigma", "2sigma", "3sigma")},
        }
        write_stats(args.out_dir, st)

    if args.per_position and pp_A is not None:
        per_position_analysis(args, args.out_dir, tok, scale, N, mean, ev, c_pp,
                              pp_A, pp_sum, n_img, dev)

    print(f"[pca] DONE in {(time.time()-t0)/60:.1f} min → {args.out_dir}", flush=True)
    print("[pca] arms_json=" + json.dumps(arms, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
