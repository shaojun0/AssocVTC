# 论文快照（2026-10-26，17 页）

本目录是**只读快照**，方便在 GitHub 上查看本轮论文改动；正文源文件在 `/userdata/paper/pp_zh/`（以其为准）。

| 文件 | 说明 |
|---|---|
| `cjc-article-v1.0_17p.pdf` | 编译产物（17 页，0 error / 0 undefined ref，overfull 仅原有 0.62 pt） |
| `fig2_step_pca.pdf` | 论文图 2：按读出步数比较像素 L1（左）与 PSNR（右），本文 224 维臂 vs 同值数逐块 PCA |
| `plot_step_pca_zs224_paper.py` | 论文版图 2 的生成脚本（matplotlib，中文字体 WenQuanYi Zen Hei） |
| `snapshot/` | 论文源文件快照：`cjc-article-v1.0.tex`、`sec3_method.tex`、`sec4_exp.tex`、`sec5_disc.tex`、`sec6_concl.tex`、`00_摘要.md` |

## 本轮（2026-10-26 晚）主要改动

1. **表 2 替换**：由「主模型四档 vs 逐块 PCA」改为**预算分层同值数对照**（1,792 / 3,584 / 7,168 / 14,336 个连续值；448 维臂 / 224 维臂 / 逐块 PCA 三行）。
2. **新增图 2**：按读出步数的 L1 与 PSNR 折线图（224 臂每步 2×224 = 448 个值；PCA 按 $r=448t/196$ 在 rank 1–40 实测值间插值，严格同值数）。
3. **删除方法章旧预算图**（读窗口递增示例，$K=64$、$T=8$ 旧 63M 协议）。
4. 摘要（中/英）、§4.1/§4.2、§5.3/§5.4、§6 **按预算分层重构**；保留反向事实（主模型 39,936 值处 27.94 dB < 逐块 PCA 35.75 dB；≥14,336 值起 PCA 反超）。

## 复现编译

```bash
cd /userdata/paper/pp_zh
export PATH=/userdata/texlive/2026/bin/aarch64-linux:$PATH
export TEXINPUTS=/userdata/texlive/2026/texmf-dist/tex/latex/ctex//:
xelatex -interaction=nonstopmode cjc-article-v1.0.tex
bibtex cjc-article-v1.0
xelatex -interaction=nonstopmode cjc-article-v1.0.tex   # ×2
```
