# setup_log — OOD 泛化评测（SAVOIAS × 容量阶梯）环境与数据准备实测日志

> 日期：2026-10-26（Asia/Shanghai）
> 评测机：`ssh -p 34314 root@connect.westb.seetacloud.com`（RTX 4080 SUPER 32G，128 核，503G RAM，`/root/autodl-tmp` 50G）
> 源模型机：`ssh -p 23556 root@connect.nma1.seetacloud.com`（2×A800-80GB；**全程只读**，未 pkill / 未写入新文件）
> 工作目录：本地 `/home/linaro/dsh/AssocVTC`（git HEAD `a24624a`，未 commit/add/checkout）；远端 `/root/AssocVTC`
> 临时数据：`/root/autodl-tmp/{ckpts,savoias,savoias_eval}`

---

## 1. 代码同步（tar over ssh + md5 对拍）

```bash
cd /home/linaro/dsh/AssocVTC
tar czf - --exclude='__pycache__' --exclude='*.pyc' *.py *.sh *.md tools feature \
  | ssh -p 34314 root@connect.westb.seetacloud.com \
      'mkdir -p /root/AssocVTC && tar xzf - -C /root/AssocVTC'
```
结果：`SYNC_OK`。关键文件 md5（本地 = 远端，逐字节一致）：

| 文件 | md5 |
|---|---|
| `eval_stepwise_linattn.py` | `00f43b9a927c0cff94d3e2312d8a1747` |
| `data_linattn.py` | `0d104af6093bbec7b967bc5421452e15` |
| `metrics_linattn.py` | `75c7deccc52d2dfb31b12d2be7a6c754` |
| `model_v2_linattn.py` | `9a5c4dc451e8821d394c8e754319f07f` |
| `tools/fetch_savoias.sh` | `1720a5a05477056407aaecb1a58c8e84` |
| `tools/plot_savoias_complexity.py` | `09bfff7f06bc4cb58937bc7373fd1da6` |

## 2. 依赖

`/root/miniconda3/bin/python` 实测：

| 包 | 版本 | 状态 |
|---|---|---|
| torch | 2.12.1+cu130 | OK（`torch.cuda.is_available()=True`） |
| torchvision | 0.27.1+cu130 | OK |
| numpy | 2.4.6 | OK |
| scipy | **缺** → `pip install scipy` → 1.18.1 | OK |
| matplotlib | 3.11.0 | OK |
| PIL | 12.2.0 | OK |

另：容器内无 `python3` 命令（仅 conda 的），`fetch_savoias.sh` 需要 `python3` →
`ln -s /root/miniconda3/bin/python3 /usr/local/bin/python3`（仅环境便利，未改仓库文件）。

## 3. SAVOIAS 数据（820 张 + gt）

**踩坑**：34314 上 `raw.githubusercontent.com` 被封（`curl` 30s 超时、http=000），
但 `codeload.github.com` 可达（≈13 MB/s）。改用整仓 tarball 绕行，再按
`tools/fetch_savoias.sh` 期望的布局整理，最后**仍然跑一遍该脚本**做校验（已存在文件全部 skip，exit 0）：

```bash
cd /root/autodl-tmp
curl -sL --max-time 600 -o savoias_master.tar.gz \
  https://codeload.github.com/esaraee/Savoias-Dataset/tar.gz/refs/heads/master   # 300,769,192 B
tar xzf savoias_master.tar.gz -C savoias_raw
R=/root/autodl-tmp/savoias_raw/Savoias-Dataset-master
mkdir -p /root/autodl-tmp/savoias/{Art,Scenes,Objects,gt}
cp -a "$R/Images/Art/."  /root/autodl-tmp/savoias/Art/
cp -a "$R/Images/Scenes/."  /root/autodl-tmp/savoias/Scenes/
cp -a "$R/Images/Objects/." /root/autodl-tmp/savoias/Objects/
cp -a "$R/Ground truth/npy/global_ranking_art.npy"     /root/autodl-tmp/savoias/gt/gt_art.npy
cp -a "$R/Ground truth/npy/global_ranking_scenes.npy"  /root/autodl-tmp/savoias/gt/gt_scenes.npy
cp -a "$R/Ground truth/npy/global_ranking_objects.npy" /root/autodl-tmp/savoias/gt/gt_objects.npy
cd /root/AssocVTC && bash tools/fetch_savoias.sh /root/autodl-tmp/savoias
```

校验结果：

- `fetch_savoias.sh` 输出：`[images] Art=421 Scenes=200 Objects=200`（Art 的 421 = 420 张 jpg + 1 个
  仓库自带的 `Images/Art/global_ranking/` **目录**，其中无图片；`scan_ood_images` 只收图片扩展名）
- 评测 json 实测：**820 张**（Art 420 / Scenes 200 / Objects 200），与预期一致
- `gt_{art,scenes,objects}.npy` 均可 `np.load`，格式 `{index: [filename, score 0-100]}`，
  n = 420 / 200 / 200；按 **文件名 stem（整数）** 与图像 join（与 `tools/plot_savoias_complexity.py` 一致）

## 4. ckpt 传输（23556 → 34314）

**踩坑 1**：经本地中转（`ssh 23556 tar | ssh 34314 tar`）只有 **~1.3 MB/s**，6.7 GB 需 ~1.5 h。
**踩坑 2**：两台机器 `authorized_keys2` 里信任的是**同一个本地 pubkey**（AutoDL 账号级），
且 23556 → 34314:34314 **TCP 直连可达** ⇒ 用本地 `ssh-agent` + `-A` 转发，
让 23556 **直推** 34314（本地只走控制流量、数据不经本地磁盘）。实测 **~10-12 MB/s，总耗时 ~15 min**。
（**没有**修改 23556 的 authorized_keys / 没有向其磁盘写任何文件。）

```bash
eval $(ssh-agent -s); ssh-add ~/.ssh/id_ed25519
ssh -A -p 23556 root@connect.nma1.seetacloud.com \
  "tar cf - -C '<src_dir>' ckpt_best.pt | \
   ssh -p 34314 root@connect.westb.seetacloud.com \
       'mkdir -p /root/autodl-tmp/ckpts/<tag> && tar xf - -C /root/autodl-tmp/ckpts/<tag>'"
```

| tag | 源路径（23556，只读） | 大小(B) | 传输耗时 | md5（源 = 目标） | torch.load 校验 |
|---|---|---:|---:|---|---|
| base63 | `/root/autodl-tmp/srdiff_n2bigstep/stdattn_k32_t16` | 756,655,771 | 64 s | `0c4a9b936fcef462c4f55fc0dd6d891a` | 63.02M, K=32, T=16, plan=fixed |
| m60 | `/root/autodl-tmp/srdiff_scale/stdattn_m60_t16` | 477,417,227 | 42 s | `762cf1208cf0eae38780ed1a343902b3` | 119.33M ✓ |
| m120 | `/root/autodl-tmp/srdiff_scale/stdattn_m120_t16` | 952,921,547 | 82 s | `a1ca850cf45a0d32b597457f5c3a8e50` | 238.20M ✓ |
| m240 | `/root/autodl-tmp/srdiff_scale/stdattn_m240_t16` | 1,926,287,179 | 457 s | `bc6f6583bda1881cf9d44ec2edb147e0` | 481.54M ✓ |
| m360 | `/root/autodl-tmp/srdiff_scale/stdattn_m360_t16` | 2,882,599,883 | 244 s | `422c1622006c2edfb02c3ff4cb0124ff` | 720.62M ✓ |

五个 ckpt 的 `cfg` 均为 `num_specials=32`、`dec_steps=1..16`（16 步）、`dec_step_plan=fixed`、
`img_size=224 / patch=16 / num_patches=196`，与 ID 参考实验一致。

## 5. 冒烟前向（8 张 SAVOIAS，`--num_images 8`）

| tag | batch_size | 结果 |
|---|---|---|
| base63 | 8 | OK，末步 L1_255≈11.20 / PSNR≈24.36（8 张抽样） |
| m60 | 8 | OK，9.29 / 25.89 |
| m120 | 8 | OK，7.73 / — |
| m240 | 8 | OK（`best_step` 集中在 15/16） |
| m360 | 8 | OK（8/8 图 `best_step=16`，无 OOM） |

全量评测 batch_size：base63/m60/m120 = 32，m240 = 16，m360 = 8（32G 卡防 OOM；m360 单次全程未 OOM）。

## 6. 全量评测（820 张，canonical 脚本，命令原样）

```bash
cd /root/AssocVTC
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 /root/miniconda3/bin/python -u eval_stepwise_linattn.py \
  --ckpt /root/autodl-tmp/ckpts/<tag>/ckpt_best.pt --data_dir_ood /root/autodl-tmp/savoias \
  --num_images 0 --batch_size <32|16|8> --num_workers 8 --domain savoias \
  --dump_per_image /root/autodl-tmp/savoias_eval/per_image_<tag>.json \
  --out /root/autodl-tmp/savoias_eval/stepwise_<tag>.json
```

实测耗时（820 张 × T=16）：base63 12 s、m60 15 s、m120 20 s、m240 ≈30 s、m360 ≈60 s；
全部 `exit=0`，日志在 `/root/autodl-tmp/savoias_eval/logs/eval_<tag>.log`。

**SSIM 补充说明**：canonical `eval_stepwise_linattn.py` 只输出 `L1_255/PSNR`，不含 SSIM。
为补齐主表 SSIM，**未改动 canonical 脚本**，另写了独立补充 pass
（`/root/autodl-tmp/savoias_eval/ssim_pass.py`，复用 `load_model`/`build_ood_dataset`/`metrics_linattn.ssim`，
`[0,1]` clamp 口径与 ID 主报告一致），第二遍前向逐图逐步算 SSIM：

```bash
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 /root/miniconda3/bin/python -u /root/autodl-tmp/savoias_eval/ssim_pass.py \
  --ckpt /root/autodl-tmp/ckpts/<tag>/ckpt_best.pt --data_dir_ood /root/autodl-tmp/savoias \
  --batch_size <32|16|8> --num_workers 8 --num_images 0 \
  --out /root/autodl-tmp/savoias_eval/per_image_ssim_<tag>.json \
  --out_summary /root/autodl-tmp/savoias_eval/ssim_<tag>.json
```

SSIM 首版 OOM（把 `B*T` 一次性展平送入 SSIM 的 conv）→ 改为**逐 step 算**后通过。
补充 pass 实测：每模型 20~90 s，5 个全部 `exit=0`；OOD 全量 SSIM 与
`ssim_<tag>.json` 的 `SSIM_last` 一致（见报告主表）。

## 7. 产物落点

- 远端：`/root/autodl-tmp/savoias_eval/{per_image_<tag>.json, stepwise_<tag>.json,
  per_image_ssim_<tag>.json, ssim_<tag>.json, logs/}`（+ 本文件）
- 远端数据/ckpt：`/root/autodl-tmp/savoias/`、`/root/autodl-tmp/ckpts/`
- 本地仓库：`doc/2026-10-26/artifacts/ood_savoias/`、`doc/2026-10-26/REPORT_ood_savoias_scale.md`、
  `tools/plot_savoias_scale.py`（分析与 4 张图由该脚本在本地生成，`--gt-dir` 指向 gt npy）

## 8. 约束遵守

- 未在 23556 上 pkill / 写新文件（仅 `tar cf -` 读 + `md5sum` 读）；其 GPU1 的 T=24 作业未受影响。
- 未 git commit / add / checkout；未改 README、`doc/2026-10-24/`、`tools/` 下既有文件。
- 34314 **未关机**。
