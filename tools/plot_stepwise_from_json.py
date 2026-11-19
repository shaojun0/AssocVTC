#!/usr/bin/env python
"""从 eval_stepwise.json 重画逐档三联图（英文标签，避免无 CJK 字体时出现豆腐块）。

用法::

    python3 tools/plot_stepwise_from_json.py \
        --json doc/2026-11-19/artifacts/v10_120f_2stage/eval_stepwise.json \
        --out  doc/2026-11-19/artifacts/v10_120f_2stage/eval_stepwise_fig.png \
        --title "v10_120f_2stage (60 val clips)"
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_ap = argparse.ArgumentParser()
_ap.add_argument("--json", required=True)
_ap.add_argument("--out", default="")
_ap.add_argument("--title", default="v10 stepwise")
_A = _ap.parse_args()

d = json.load(open(_A.json, encoding="utf-8"))
c = d["curves"]
n = int(c["n_steps"])
xs = np.arange(1, n + 1)
P = np.asarray(c["psnr_all_mean"], dtype=float)
PR = np.asarray(c["psnr_read_mean"], dtype=float)
L = np.asarray(c["l1_all_mean"], dtype=float)
LR = np.asarray(c["l1_read_mean"], dtype=float)
ZC = np.asarray(c["zcls_l1_mean"], dtype=float)
SP = np.asarray(c["spec_l1_mean"], dtype=float)
TF = float(c["psnr_tok_full_mean"])

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].plot(xs, P, "-o", ms=3, label="whole clip (incl. unread frames)")
ax[0].plot(xs, PR, "-s", ms=3, label="read frames only (training loss view)")
ax[0].axhline(TF, ls="--", c="gray", label=f"tokenizer full read {TF:.2f} dB")
ax[0].set_xlabel("z steps read (rate budget)")
ax[0].set_ylabel("PSNR (dB)")
ax[0].set_title(f"stepwise PSNR — {_A.title}")
ax[0].grid(alpha=.3)
ax[0].legend(fontsize=7)

ax[1].plot(xs, L, "-o", ms=3, label="whole clip latent L1")
ax[1].plot(xs, LR, "-s", ms=3, label="read frames latent L1")
ax[1].set_xlabel("z steps read")
ax[1].set_ylabel("latent L1")
ax[1].set_title("stepwise latent L1")
ax[1].grid(alpha=.3)
ax[1].legend(fontsize=7)

ax[2].plot(xs, ZC, "-o", ms=3, label="z_cls row error")
ax[2].plot(xs, SP, "-s", ms=3, label="special rows error")
ax[2].set_xlabel("z steps read")
ax[2].set_ylabel("latent L1")
ax[2].set_title("error decomposition (z_cls vs special)")
ax[2].grid(alpha=.3)
ax[2].legend(fontsize=7)
fig.tight_layout()
out = _A.out or _A.json.replace(".json", "_fig.png")
fig.savefig(out, dpi=130)
print(f"[fig] {out}")
