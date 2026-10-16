"""
AssocVTC Phase 1 v2 — register 式推理测试（像素目标版, 2026-08-27 起）
=================================================
架构（model_v2.py, register 式唯一路径）: DINOv2-large + register
    specials(K) 直接拼进输入序列 → OutputQueryDecoder（**顺序循环**: 每采样步
    只读自己那块 z_s 切片）→ PixelHead → 像素 patch 预测。K = num_specials 与
    N = num_patches 解耦（K 由训练侧最终采样步集推导/显式指定, 见 train_v2.py
    与 model_v2.py derive_num_specials）。

测试项（像素目标 = 最终判据）:
    1) 全量重建像素 L1（归一化空间 + 反归一化 0-255 空间双口径）——
       对照: 全图平均色 baseline / 每 patch 平均色 baseline。
       像素 L1 必须显著优于"平均色"才有还原意义（特征空间 L1 是假象,
       已证实特征目标退化）。
    2) 渐进重建曲线（2026-08-31 用户需求; 2026-09-15 起口径 = **直接预测**）——
        每个采样步的输出 Y_t **各自直接**过 PixelHead 预测整图（无累加/集成;
        旧的累加口径 Σ_{t≤n} Y_t 已随 model_v2.py 的累加路径一并删除）:
       Y_pix[:, n] = PixelHead(Y_n)。对每步 n 度量像素 L1(Y_pix[:, n],
       target_pix), 得到"累积步数越多重建越精"的渐进曲线。

2026-08-31（分块读窗口 + 边界实验对齐）:
    · decoder 读窗口从 KV 因果前缀改为**分块**: 步 t 只读自己那块 z_s
      （块号 ⌊√t⌋）; 默认采样计划 = square_block_starts
      （块起点 = 平方数, 每块一步, 步数 = ⌊√N⌋）;
    · --slice_start/--slice_end 可选挑选分块子区间（默认 None = 全部分块）;
    · --decoder_steps 越界校验: 0 <= s <= N（K 的最终校验交给模型）。

2026-09-02（num_specials 对齐; 修复与 model_v2.py 接口脱节）:
    · 删除 --reencoder_depth / --register_specials（register 式唯一路径,
      model_v2.py 无这些接口）;
    · K 复现训练值: **优先读 model_info.json 的 num_specials**; 没有则
      --num_specials CLI（默认 0=auto, 按与训练一致的 slice/decoder_steps
      自动推导, 公式同 derive_num_specials）; 都没有 = 全量默认 K=N。
      K 必须与 final_model.pt 权重形状一致, 否则 strict load 直接崩。

用法:
    python infer_v2_test.py \
        --data_dir /root/autodl-tmp/construction_site \
        --dino_dir /root/autodl-tmp/models/dinov2-large \
        --final_model output/phase1_v2_pixelfp32/final_model.pt \
        --output output/phase1_v2_pixelfp32/infer_test.json
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Dinov2Model

from data_v2 import ParquetImageDataset, V2Collator, DINO_MEAN, DINO_STD
from model_v2 import SRPhase1V2, patches_to_image, step_windows


def parse_args():
    p = argparse.ArgumentParser(description="infer test (pixel target) for OutputQueryDecoder v2")
    p.add_argument("--data_dir", required=True, help="parquet 目录(test-*.parquet)")
    p.add_argument("--dino_dir", default="models/dinov2-large")
    p.add_argument("--final_model", required=True, help="训练好的权重(final_model.pt)")
    p.add_argument("--output", default="output/phase1_v2_pixelfp32/infer_test.json")
    p.add_argument("--model_input", default="448x252")
    p.add_argument("--canvas", default="1600x900")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="只用前 N 条 test(0=全量)")
    p.add_argument("--num_specials", type=int, default=0,
                   help="register/specials 数 K: 0=自动（按训练一致的 "
                        "slice/decoder_steps 推导, 公式同训练）; >0=显式 K。"
                        "model_info.json 有 num_specials 字段时以它为准")
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--decoder_depth", type=int, default=2,
                   help="OutputQueryDecoder 的 TransformerDecoder 层数(与训练一致)")
    p.add_argument("--stack_dim", type=int, default=0,
                   help="OutputQueryDecoder 的 self.stack 的 d_model(与训练一致); "
                        "0=与 dim 相同。model_info.json 有 stack_dim 字段时以它为准")
    p.add_argument("--decoder_dropout", type=float, default=0.0,
                   help="self.stack 的 dropout(与训练一致; eval 推理下不生效, 但构造须对齐)")
    p.add_argument("--slice_start", type=int, default=None,
                   help="可选挑选分块起点索引(与训练 --slice_start 一致); 默认 None = 全部分块")
    p.add_argument("--slice_end", type=int, default=None,
                   help="可选挑选分块终点索引(与训练 --slice_end 一致); 默认 None = 全部分块")
    p.add_argument("--decoder_steps", default=None,
                   help="必须与训练一致(逗号分隔); 默认 square_block_starts(N) (分块起点=平方数)")
    p.add_argument("--step_plan", default="square", choices=["square", "fixed"],
                   help="解码器读窗口计划(必须与训练一致): square=步值 k²（历史口径）; "
                        "fixed=步值自然数、每步固定读 block 个 z_s（2026-09-23 fixw4 口径）。"
                        "model_info.json 有 step_plan 字段时以它为准")
    p.add_argument("--block", type=int, default=0,
                   help="--step_plan fixed 时每步读的 z_s 个数(必须与训练一致)")
    # ── 2026-09-22 新增: 论文口径指标（PLAN_paper_experiments.md §3 批 0）──
    p.add_argument("--no_ms_ssim", action="store_true",
                   help="跳过 MS-SSIM（自实现, 5 尺度 Gaussian/Y 通道）; 默认算")
    p.add_argument("--no_per_image", action="store_true",
                   help="不落逐图逐 step 数组（默认落, 供 E4b 自适应早停分析）")
    # 注: 解码器是顺序循环（唯一路径, 见 model_v2.py OutputQueryDecoder）;
    # 2026-09-15 之前并行路径的 --recurrent* 开关已删除。
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
# 评测指标 —— PSNR / MS-SSIM（自实现, 不引入第三方依赖）
# ═══════════════════════════════════════════════════════════════
# 为什么必须补(PLAN §3.2): 文献表只有 PSNR/MS-SSIM/LPIPS/FID, 没有 L1 这一列;
# L1 与 PSNR 无函数关系 ⇒ 不补就一行都同不了表。
_MS_SSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def _gauss_1d(win: int = 11, sigma: float = 1.5, device=None, dtype=None):
    x = torch.arange(win, device=device, dtype=dtype) - (win - 1) / 2.0
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _gauss_blur(x, g):
    """(B,1,H,W) 与可分离 Gaussian 做 valid 卷积（win×win 方核）。"""
    win = g.numel()
    k = (g[:, None] * g[None, :]).view(1, 1, win, win)
    return F.conv2d(x, k, padding=0)


def _ssim_cs(x, y, g):
    """x, y: (B,1,H,W) 0-255 ⇒ (ssim, cs) 的逐位置图。"""
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx, my = _gauss_blur(x, g), _gauss_blur(y, g)
    mxx, myy, mxy = mx * mx, my * my, mx * my
    sxx = _gauss_blur(x * x, g) - mxx
    syy = _gauss_blur(y * y, g) - myy
    sxy = _gauss_blur(x * y, g) - mxy
    cs = (2 * sxy + c2) / (sxx + syy + c2)
    ssim = ((2 * mxy + c1) * (2 * sxy + c2)) / ((mxx + myy + c1) * (sxx + syy + c2))
    return ssim, cs


def ms_ssim(gt, rec):
    """gt, rec: (B,3,H,W) 0-255 float ⇒ MS-SSIM (B,)。

    口径: BT.601 亮度 Y → 5 尺度 → 每尺度 separable Gaussian(11, σ=1.5) →
    逐图空间均值 → 权重 [0.0448,0.2856,0.3001,0.2363,0.1333]（Wang et al. 2003,
    与 pytorch_msssim.ms_ssim 同口径）。要求 min(H,W) >= 176（5 尺度后仍 ≥11）。
    """
    w = torch.tensor([0.299, 0.587, 0.114], device=gt.device, dtype=gt.dtype)
    x = (gt * w.view(1, 3, 1, 1)).sum(1, keepdim=True)
    y = (rec * w.view(1, 3, 1, 1)).sum(1, keepdim=True)
    n = ms_ssim_n_scales(x.shape[-2], x.shape[-1])
    if n == 0:
        return torch.full((gt.shape[0],), float("nan"), device=gt.device)
    wts = np.asarray(_MS_SSIM_WEIGHTS[:n], np.float64)
    wts = wts / wts.sum()          # 尺度数不足时按比例重归一
    g = _gauss_1d(11, 1.5, gt.device, gt.dtype)
    out = None
    for i in range(n):
        ssim, cs = _ssim_cs(x, y, g)
        wt = float(wts[i])
        if i < n - 1:
            v = cs.mean(dim=(1, 2, 3)).clamp(min=1e-6) ** wt
            x, y = F.avg_pool2d(x, 2), F.avg_pool2d(y, 2)
        else:
            v = ssim.mean(dim=(1, 2, 3)).clamp(min=1e-6) ** wt
        out = v if out is None else out * v
    return out


def ms_ssim_n_scales(H: int, W: int, win: int = 11) -> int:
    """本图尺寸能用几个 MS-SSIM 尺度（标准 5 尺度在 224×126 上放不下）。

    每级做 valid Gaussian 卷积, 要求该级 min(H,W) ≥ win。224×126 → 126,63,31,15
    ⇒ n=4；448×252 → 252,126,63,31,15 ⇒ n=5（与 pytorch_msssim 一致）。
    尺度数不足时权重按比例重归一（论文里必须声明用了几个尺度）。
    """
    m = min(H, W)
    n = 0
    while n < len(_MS_SSIM_WEIGHTS) and (m >> n) >= win:
        n += 1
    return n


def psnr_from_mse(mse):
    """10·log10(255²/MSE)；mse 可为标量或 np 数组。"""
    mse = np.maximum(np.asarray(mse, dtype=np.float64), 1e-12)
    return 10.0 * np.log10(255.0 ** 2 / mse)


def _patch_to_img(pix_patches, H, W):
    """(B,N,588) 归一化像素 patch → (B,H,W,3) 反归一化 0-255 numpy 图像。

    布局与反归一化统一走 model_v2.patches_to_image（target_pix 布局的唯一反
    变换, 见其 docstring 与 model_v2.py 自检 §1b 的往返断言）。**不要在消费方
    手写 reshape**: 2026-09-15 前这里把 patch 向量当 "(14,14,3) 通道在后" 读,
    与 decode 的通道优先 (C,14,14) 不符 ⇒ 每个 14×14 patch 内部像素被打乱。
    """
    return patches_to_image(pix_patches, H, W, DINO_MEAN, DINO_STD) \
        .cpu().numpy()


def main():
    args = parse_args()
    W, H = (int(v) for v in args.model_input.lower().split("x"))
    num_patches = (W // 14) * (H // 14)
    PATCH_PX = 14 * 14 * 3

    steps = None
    if args.decoder_steps:
        steps = [int(s) for s in args.decoder_steps.split(",") if s.strip()]
        # 采样时刻 t ∈ [0, N]; t ≤ K ≤ N 的最终校验由模型构造时完成
        # （与 train_v2.py 相同的解析/校验逻辑）
        assert steps and all(0 <= s <= num_patches for s in steps), \
            f"decoder_steps 越界: {steps} (N={num_patches})"

    # ── model_info.json: **结构超参一律以训练侧记录为准** ──
    # 训练侧把 num_specials / decoder_depth / heads / mlp_ratio / slice_start /
    # slice_end / decoder_steps / stack_dim / decoder_dropout 写在
    # output_dir/model_info.json。其中 heads / mlp_ratio 不改变权重形状 ⇒ 传错时
    # strict load 不报错只会静默算错; decoder_depth / stack_dim 传错则形状不符直接崩。
    # 故构造模型前先用 model_info 覆盖 CLI。
    info_path = os.path.join(os.path.dirname(args.final_model), "model_info.json")
    train_info = None
    if os.path.exists(info_path):
        with open(info_path) as f:
            train_info = json.load(f)

    def _pick(name, cli, default):
        """model_info.json 优先; CLI 与之不一致且非默认时告警, 仍以 model_info 为准。"""
        if train_info is not None and name in train_info:
            val = train_info[name]
            if cli != val and cli != default:
                print(f"[warn] model_info.json 记录 {name}={val!r}, 与 "
                      f"--{name}={cli!r} 不一致: 以 model_info 为准")
            if val is not None:
                return val
        return cli

    decoder_depth = int(_pick("decoder_depth", args.decoder_depth, 2))
    heads = int(_pick("heads", args.heads, 8))
    mlp_ratio = float(_pick("mlp_ratio", args.mlp_ratio, 4.0))
    slice_start = _pick("slice_start", args.slice_start, None)
    slice_end = _pick("slice_end", args.slice_end, None)
    stack_dim = int(_pick("stack_dim", args.stack_dim, 0))
    decoder_dropout = float(_pick("decoder_dropout", args.decoder_dropout, 0.0))
    # 读窗口计划: fixw4 产物必须回放 step_plan=fixed + block, 否则 step_windows
    # 会按 square 公式解释 [1..144] 的步值 ⇒ 静默读错窗口、指标全错。
    step_plan = _pick("step_plan", args.step_plan, "square")
    block = int(_pick("block", args.block, 0))
    if step_plan == "fixed" and block < 1:
        raise SystemExit("step_plan=fixed 需要 block ≥1（model_info.json 或 --block）")
    if steps is None and train_info is not None and "decoder_steps" in train_info:
        steps = [int(s) for s in train_info["decoder_steps"]]
    # num_specials(K) 解析: ① model_info.json 优先; ② --num_specials CLI;
    # ③ 都没有 → None = 全量默认 K=N
    num_specials = None
    if train_info is not None and "num_specials" in train_info:
        num_specials = int(train_info["num_specials"])
        if args.num_specials and args.num_specials != num_specials:
            print(f"[warn] model_info.json 记录 num_specials={num_specials}, 与 "
                  f"--num_specials={args.num_specials} 不一致: 以 model_info 为准")
    elif args.num_specials:
        num_specials = args.num_specials
    if train_info is not None and "num_specials" not in train_info:
        # 旧 register 训练产物（K=N=num_patches, 无 num_specials 字段）:
        # 若 slice 非默认, 自动推导的 K 会与旧权重不符 → strict load 崩
        print(f"[info] model_info.json 无 num_specials 字段（旧 register 产物, "
              f"K=N={num_patches}）: 若 strict load 形状不符, 请显式 "
              f"--num_specials {num_patches}")

    # ── 模型: 训练好的重建权重 ──
    dino = Dinov2Model.from_pretrained(args.dino_dir)
    if getattr(dino.config, "use_mask_token", False):
        dino.config.use_mask_token = False
        del dino.embeddings.mask_token
    # 解码器: 当前架构只有顺序循环一条路径, 没有开关可对齐。
    # ⚠️ checkpoint 兼容性（2026-09-15 更正, 与旧注释相反）: 循环版**不新增
    # 任何参数**（rec_* 随并行路径的开关一起删掉了）, 与并行时代默认配置的
    # state_dict 逐 key 逐形状完全相同（实测 44 keys 全等）⇒ 2026-09-15 之前
    # 训出的 final_model.pt 用本代码 strict load **不会报错**, 会按新语义
    # （循环 + carry detach + 直预 loss）静默算错、画出/量错的图。
    # 本文件与 model_info.json 目前都无法自动区分这两种产物, 判据只能靠人工:
    # 产物目录的 args.json/model_info.json 里若没有本轮新增字段（如
    # "decoder_steps" 之外看不到循环语义）, 一律先确认它是在哪个 commit 训的;
    # 要复现并行产物请用 git 取回当时的 model_v2.py, 不要拿旧权重在本代码上推理。
    model = SRPhase1V2(dinov2=dino, num_patches=num_patches,
                       dim=dino.config.hidden_size,
                       heads=heads, mlp_ratio=mlp_ratio,
                       decoder_steps=steps,
                       decoder_depth=decoder_depth,
                       skip_steps=slice_start,
                       max_steps=slice_end,
                       num_specials=num_specials,
                       stack_dim=stack_dim,
                       decoder_dropout=decoder_dropout,
                       step_plan=step_plan, block=block)
    sd = torch.load(args.final_model, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval().cuda()
    T_steps = model.decoder.steps
    print(f"[model] loaded {args.final_model}: N={num_patches}, "
          f"K(num_specials)={model.num_specials}, "
          f"plan={step_plan}" + (f"(block={block})" if step_plan == "fixed" else "")
          + f", decoder 采样 {len(T_steps)} 步 {T_steps[:6]}...{T_steps[-3:]}")

    # ── model_info.json 对齐提示（加载后完整对比, 不强制）──
    if train_info is not None:
        mism = []
        if ("num_specials" in train_info
                and int(train_info["num_specials"]) != model.num_specials):
            mism.append(f"num_specials: 训练 {train_info['num_specials']} "
                        f"!= 推理 {model.num_specials}")
        if ("decoder_steps" in train_info
                and list(train_info["decoder_steps"]) != T_steps):
            mism.append(f"decoder_steps: 训练 {train_info['decoder_steps']} "
                        f"!= 推理 {T_steps}")
        if ("layer_tap" in train_info or "tap_groups" in train_info):
            mism.append("layer_tap/tap_groups: 该实验已于 2026-09-23 从 main 移除"
                        "（旧产物请用当时 commit 的代码推理）")
        if mism:
            print(f"[warn] 推理参数与训练侧 model_info.json 不一致 ({info_path}):")
            for m in mism:
                print(f"    - {m}")
        else:
            print(f"[ok] 推理参数与训练侧 model_info.json 对齐 ({info_path})")

    # ── 数据: test 分片, 与训练同预处理（1600:900 画布 → 448x252）──
    test_files = sorted(glob.glob(os.path.join(args.data_dir, "test-*.parquet")))
    assert test_files, f"无 test-*.parquet in {args.data_dir}"
    coll = V2Collator(model_size=(W, H), canvas=(1600, 900), return_mask=True)
    ds = ParquetImageDataset(test_files, limit=args.limit)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=coll)
    n_total = len(ds)
    print(f"[data] {n_total} 条 test")

    # ── 前向: 全量 test, 像素 L1（归一化空间 + 0-255 空间）──
    norm_sum, norm_sq, pix_sum, pix_sq, n = 0.0, 0.0, 0.0, 0.0, 0
    step_pix_sum = np.zeros(len(T_steps), np.float64)   # 每采样步像素 L1 (0-255)
    step_pix_sq = np.zeros(len(T_steps), np.float64)
    # ── 2026-09-22 新增: PSNR / MS-SSIM / 内容区口径（PLAN 批 0）──
    step_mse_sum = np.zeros(len(T_steps), np.float64)   # 全画布 MSE
    step_l1_sum_c = np.zeros(len(T_steps), np.float64)  # 内容区 L1（剔 letterbox）
    step_mse_sum_c = np.zeros(len(T_steps), np.float64)  # 内容区 MSE
    step_ssim_sum = np.zeros(len(T_steps), np.float64)  # MS-SSIM
    psnr_rows: list = []                                # 每 batch 逐图 PSNR
    content_frac_sum = 0.0
    # 平凡基线: 输出恒为画布填充色 (fill=(0,0,0)) ⇒ 误差就是 gt 本身。
    # 用来回答 PLAN §5 第 2 条: RD 曲线最左端是不是"把黑边填对"撑起来的。
    triv_l1_sum = triv_mse_sum = 0.0
    triv_l1_c_sum = triv_mse_c_sum = 0.0
    do_ssim = not args.no_ms_ssim
    do_per_image = not args.no_per_image
    per_image: list = [] if do_per_image else []
    D_DIM = dino.config.hidden_size
    PX = H * W
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            x = batch["pixel_values"].cuda()            # (B,3,H,W) 归一化
            B, C, Hh, Ww = x.shape
            out = model(x)                              # 同一 forward（两种模式通用）
            F_pix = out["F_hat"]                        # (B,N,588) 最后一步的直接预测
            Y_pix = out["Y_pix"]                        # (B,|T|,N,588) 每采样步
            target = out["target_pix"]                  # (B,N,588)
            # 归一化空间 L1
            l1_norm = (F_pix - target).abs().mean(dim=(1, 2))          # (B,)
            norm_sum += l1_norm.sum().item()
            norm_sq += (l1_norm ** 2).sum().item()
            # 0-255 空间: 反归一化 (patch 级, 与 pixel_recon_check 同口径)
            gt_255 = _patch_to_img(target, Hh, Ww)      # (B,H,W,3)
            recon_255 = _patch_to_img(F_pix, Hh, Ww)
            l1_pix = np.abs(recon_255 - gt_255).mean(axis=(1, 2, 3))   # (B,)
            pix_sum += l1_pix.sum().item()
            pix_sq += (l1_pix ** 2).sum().item()
            # 内容区 mask: True=真内容, False=letterbox padding（fill=(0,0,0)）
            mask = batch.get("content_mask")
            mask_np = mask.numpy() if mask is not None else None       # (B,H,W) bool
            if mask_np is not None:
                content_frac_sum += float(mask_np.mean()) * B
            # 平凡基线（与步数无关, 每 batch 算一次）
            triv_l1_sum += float(np.abs(gt_255).mean(axis=(1, 2, 3)).sum())
            triv_mse_sum += float((gt_255.astype(np.float64) ** 2)
                                  .mean(axis=(1, 2, 3)).sum())
            if mask_np is not None:
                m0 = mask_np[..., None].astype(np.float64)
                d0 = (np.maximum(mask_np.sum(axis=(1, 2)), 1).astype(np.float64)
                      * gt_255.shape[-1])
                triv_l1_c_sum += float(((np.abs(gt_255) * m0)
                                        .sum(axis=(1, 2, 3)) / d0).sum())
                triv_mse_c_sum += float((((gt_255.astype(np.float64) ** 2) * m0)
                                         .sum(axis=(1, 2, 3)) / d0).sum())
            paths = batch.get("image_path")
            base = len(per_image)
            if do_per_image:
                for j in range(B):
                    per_image.append({
                        "path": paths[j] if paths else None,
                        "content_frac": (float(mask_np[j].mean())
                                         if mask_np is not None else 1.0),
                        "step_l1": [], "step_mse": [], "step_psnr": [],
                        "step_l1_content": [], "step_mse_content": [],
                        "step_ms_ssim": [],
                    })
            gt_t = (torch.from_numpy(gt_255).permute(0, 3, 1, 2).contiguous()
                    .float().cuda()) if do_ssim else None
            row_psnr = np.zeros((B, len(T_steps)), np.float64)
            # 每采样步像素 L1/MSE/PSNR/MS-SSIM (0-255)
            for i in range(len(T_steps)):
                step_img = _patch_to_img(Y_pix[:, i], Hh, Ww)
                diff = step_img - gt_255
                sl1 = np.abs(diff).mean(axis=(1, 2, 3))                # (B,)
                smse = (diff ** 2).mean(axis=(1, 2, 3))                # (B,)
                step_pix_sum[i] += sl1.sum().item()
                step_pix_sq[i] += (sl1 ** 2).sum().item()
                step_mse_sum[i] += smse.sum().item()
                row_psnr[:, i] = psnr_from_mse(smse)
                l1c = msec = None
                if mask_np is not None:
                    m = mask_np[..., None].astype(np.float64)
                    # 分母必须是「内容像素数 × 通道数」（与全画布 .mean((1,2,3)) 同口径）
                    den = (np.maximum(mask_np.sum(axis=(1, 2)), 1).astype(np.float64)
                           * diff.shape[-1])
                    l1c = (np.abs(diff) * m).sum(axis=(1, 2, 3)) / den
                    msec = ((diff ** 2) * m).sum(axis=(1, 2, 3)) / den
                    step_l1_sum_c[i] += l1c.sum().item()
                    step_mse_sum_c[i] += msec.sum().item()
                s_np = None
                if do_ssim:
                    rec_t = (torch.from_numpy(step_img).permute(0, 3, 1, 2)
                             .contiguous().float().cuda())
                    s = ms_ssim(gt_t, rec_t)
                    step_ssim_sum[i] += float(torch.nan_to_num(s, nan=0.0).sum())
                    s_np = s.detach().cpu().numpy()
                if do_per_image:
                    for j in range(B):
                        d = per_image[base + j]
                        d["step_l1"].append(float(sl1[j]))
                        d["step_mse"].append(float(smse[j]))
                        d["step_psnr"].append(float(row_psnr[j, i]))
                        if l1c is not None:
                            d["step_l1_content"].append(float(l1c[j]))
                            d["step_mse_content"].append(float(msec[j]))
                        if s_np is not None:
                            d["step_ms_ssim"].append(float(s_np[j]))
            psnr_rows.append(row_psnr)
            n += B
            if (bi + 1) % 10 == 0 or (bi + 1) == len(loader):
                print(f"  ... {n}/{n_total} ({time.time() - t0:.0f}s)", flush=True)

    norm_mean = norm_sum / n
    pix_mean = pix_sum / n
    pix_std = np.sqrt(max(pix_sq / n - pix_mean ** 2, 0.0))
    step_pix_mean = step_pix_sum / n
    step_pix_std = np.sqrt(np.maximum(step_pix_sq / n - step_pix_mean ** 2, 0.0))
    # ── 2026-09-22 新增口径 ──
    step_mse = step_mse_sum / n
    step_psnr = psnr_from_mse(step_mse)     # 标准口径: 由「平均 MSE」反算
    psnr_rows = (np.concatenate(psnr_rows, axis=0) if psnr_rows
                 else np.zeros((0, len(T_steps))))
    step_psnr_img_mean = psnr_rows.mean(axis=0)
    step_psnr_img_std = psnr_rows.std(axis=0)
    step_l1_c = step_l1_sum_c / n
    step_mse_c = step_mse_sum_c / n
    step_psnr_c = psnr_from_mse(step_mse_c)
    step_ssim = step_ssim_sum / n
    content_frac = content_frac_sum / n if n else 1.0
    triv_l1 = triv_l1_sum / n
    triv_psnr = float(psnr_from_mse(triv_mse_sum / n))
    triv_l1_c = triv_l1_c_sum / n
    triv_psnr_c = float(psnr_from_mse(triv_mse_c_sum / n))
    # 每步实际读入解码器的 token 数 = t 个 register + 首步另读的 z_cls（平方块口径;
    # 逐层 tap 已于 2026-09-23 从 main 移除）⇒ 偏移恒为 1; 这决定 bpp 轴。
    TOK_OFF = 1
    bpp_of = lambda t: (t + TOK_OFF) * D_DIM * 1.0 / PX   # β=1 bit/dim 估计
    # 实际累计读入（2026-09-24）: 步 t 读 A[:, lo:hi] ⇒ 累计列数 = hi+1（含 z_cls）。
    # square 计划下 = (k+1)²; fixed 计划下 = min(k·block, K)。fixed 的**名义** t+1
    # 比实际小 ~block 倍 ⇒ 画 RD 必须用 bpp_actual（见 REPORT_fixw4_plan.md §5.5）。
    _wins = step_windows(model.num_specials, T_steps, step_plan, block)
    win = [hi - lo + 1 for lo, hi in _wins]
    cum_read = [hi + 1 for _, hi in _wins]
    bpp_actual_of = lambda c: c * D_DIM * 1.0 / PX

    print(f"\n[full] 全量重建像素 L1 (归一化空间) = {norm_mean:.6f}")
    print(f"[full] 全量重建像素 L1 (0-255 空间) = {pix_mean:.2f} ± {pix_std:.2f}")
    print(f"[full] 全画布 MSE(0-255) = {(recon_255 - gt_255).__abs__().mean():.4f} "
          f"(仅末批) | PSNR = {psnr_from_mse(((recon_255 - gt_255) ** 2).mean()):.2f} dB")
    print(f"[mask] 内容区占比 = {content_frac * 100:.2f}% "
          f"(padding {(1 - content_frac) * 100:.2f}%, fill=(0,0,0))")
    print(f"[triv] 平凡基线(全黑画布): 全画布 L1={triv_l1:.2f} PSNR={triv_psnr:.2f} dB | "
          f"内容区 L1={triv_l1_c:.2f} PSNR={triv_psnr_c:.2f} dB")
    print(f"       ⚠️ 最左端 (t=1) 的全画布 PSNR={step_psnr[0]:.2f} dB 必须显著高于 "
          f"{triv_psnr:.2f} dB 才算真重建, 否则曲线左端是 padding 撑的")
    print(f"       参照(旧实验): 全图平均色≈61, 每patch平均色≈?, 质心基线见 pixel_recon_check")
    print(f"\n[steps] 渐进曲线 ({len(T_steps)} 步, 0-255; plan={step_plan}"
          + (f"(block={block})" if step_plan == "fixed" else "")
          + f"; bppN = 名义 (t+{TOK_OFF})·{D_DIM}/{PX} @β=1, "
          f"bppA = 实际累计读入·{D_DIM}/{PX}（fixed 计划必须看 bppA）):")
    print(f"    {'t':>4}{'read':>6}{'bppN':>7}{'bppA':>7}{'L1':>9}{'PSNR':>8}{'MS-SSIM':>9}"
          f"{'L1内容区':>10}{'PSNR内容区':>11}")
    for i, t in enumerate(T_steps):
        print(f"    {t:>4}{cum_read[i]:>6}{bpp_of(t):>7.3f}"
              f"{bpp_actual_of(cum_read[i]):>7.3f}"
              f"{step_pix_mean[i]:>9.3f}{step_psnr[i]:>8.2f}{step_ssim[i]:>9.4f}"
              f"{step_l1_c[i]:>10.3f}{step_psnr_c[i]:>11.2f}")
    head = step_pix_mean[:min(4, len(step_pix_mean))]
    tail = step_pix_mean[max(0, len(step_pix_mean) - 4):]
    print(f"[steps] 前段(早步) {head.mean():.2f} | 后段(晚步) {tail.mean():.2f} | "
          f"首步/末步 = {step_pix_mean[0]:.2f}/{step_pix_mean[-1]:.2f}")
    print(f"[time] {(time.time() - t0):.0f}s | {n} 图")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    payload = {
        "n": n, "num_patches": num_patches,
        "num_specials": model.num_specials, "input": [W, H],
        "decoder_steps": T_steps,
        "full_norm_l1": float(norm_mean),
        "full_pixel_l1_255": float(pix_mean), "full_pixel_std_255": float(pix_std),
        "step_pixel_l1_255": [float(v) for v in step_pix_mean],
        "step_pixel_std_255": [float(v) for v in step_pix_std],
        "head_mean": float(head.mean()), "tail_mean": float(tail.mean()),
        "shortest_l1": float(step_pix_mean[0]), "longest_l1": float(step_pix_mean[-1]),
        "time_s": round(time.time() - t0, 1),
        # ── 2026-09-22 新增（PLAN 批 0: rate + PSNR + 逐图）──
        "schema_version": 2,
        "step_mse_255": [float(v) for v in step_mse],
        "step_psnr": [float(v) for v in step_psnr],
        "step_psnr_img_mean": [float(v) for v in step_psnr_img_mean],
        "step_psnr_img_std": [float(v) for v in step_psnr_img_std],
        "step_ms_ssim": [float(v) for v in step_ssim],
        "step_l1_content_255": [float(v) for v in step_l1_c],
        "step_mse_content_255": [float(v) for v in step_mse_c],
        "step_psnr_content": [float(v) for v in step_psnr_c],
        "content_frac_mean": float(content_frac),
        "trivial_black_l1_255": float(triv_l1),
        "trivial_black_psnr": triv_psnr,
        "trivial_black_l1_content_255": float(triv_l1_c),
        "trivial_black_psnr_content": triv_psnr_c,
        "ms_ssim_available": bool(do_ssim),
        "ms_ssim_scales": int(ms_ssim_n_scales(H, W)),
        "bpp_beta": 1.0,
        "bpp_px": int(PX),
        "step_bpp_beta1": [float(bpp_of(t)) for t in T_steps],
        "step_tokens": [int(t + TOK_OFF) for t in T_steps],
        # ── 2026-09-24 新增: fixed 计划的实际读窗口/累计读入（bpp_actual）──
        "step_plan": step_plan, "block": int(block),
        "step_win": [int(v) for v in win],
        "step_cum_read": [int(v) for v in cum_read],
        "step_bpp_actual_beta1": [float(bpp_actual_of(c)) for c in cum_read],
        "canvas_fill": [0, 0, 0],
        "metrics_note": ("PSNR = 10log10(255^2/mean_MSE) 由平均 MSE 反算（压缩界标准口径）；"
                         "step_psnr_img_mean = 逐图 PSNR 的均值。step_bpp_beta1 为**名义**估计 "
                         "(t+tok_off)·D·β/(H·W)（tok_off = 1）；step_bpp_actual_beta1 为**实际**"
                         "累计读入 hi+1 列（fixed 计划下名义值小 ~block 倍，画 RD 用 actual），"
                         "均未量化/熵编码，真实 bpp 见 PLAN §4 E3。"
                         "内容区口径用 content_mask 剔除 letterbox padding。"),
    }
    if do_per_image:
        payload["per_image"] = per_image
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[save] {args.output}")


if __name__ == "__main__":
    main()
