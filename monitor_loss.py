#!/usr/bin/env python3
"""vlm8b 损失监控 + 平台期判定（续训期间定时检查用）。

数据来源: `<output-dir>/per_budget_ce.jsonl` —— 每个 logging step（=20 步）
追加一行，含 `per_budget_ce[k=1..8]`（teacher-forcing、仅 answer token 的 CE，
已做因果 shift、无目标泄露）、`loss`、`learning_rate`、`grad_norm`。

为什么看 `per_budget_ce` 而不是日志里的 `loss`：两者尺度不同（loss 是另一个
归一口径），而 `per_budget_ce` 与 `eval.py` 报的 CE 是同一个口径，可以直接和
run1 终点 / 评测值比较。

判定逻辑（训练完成 `--min-steps` 步后）:
  * 最近 `--window` 步 CE 均值 vs 之前同样长度窗口的 CE 均值；
  * 相对改善 < `--rel-tol`（默认 2%）且绝对变化 < `--abs-tol`（默认 0.010）
    且 grad_norm 没有明显上升  →  PLATEAU（建议动 lr）
  * CE 变差 > `--rel-tol` 或 grad_norm > `--gn-mult`× 早期基线  →  DIVERGING
  * 其余  →  IMPROVING（继续跑，不动超参）

退出码: 0 = IMPROVING/数据不足, 1 = PLATEAU, 2 = DIVERGING。
"""
import argparse
import json
import statistics
from typing import List, Optional, Tuple

CE_KEY = "per_budget_ce"
LOSS_KEY = "loss"


def load_rows(path: str) -> List[dict]:
    """读 JSONL，跳过坏行（训练进程仍在写时会读到半行）。"""
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if CE_KEY not in record:
                continue
            rows.append(record)
    return rows


def ce_of(row: dict) -> Optional[float]:
    values = row.get(CE_KEY) or []
    return sum(values) / len(values) if values else None


def window_mean(rows: List[dict], lo: float, hi: float) -> Optional[float]:
    values = [ce_of(r) for r in rows if lo <= r["step"] < hi and ce_of(r) is not None]
    return sum(values) / len(values) if values else None


def loss_of(row: dict) -> Optional[float]:
    value = row.get(LOSS_KEY)
    return float(value) if value is not None else None


def loss_window(rows: List[dict], lo: float, hi: float) -> Optional[float]:
    values = [loss_of(r) for r in rows if lo <= r["step"] < hi and loss_of(r) is not None]
    return sum(values) / len(values) if values else None


def slope_per_100(rows: List[dict], lo: float) -> Optional[float]:
    """窗口内 CE 对 step 的最小二乘斜率（每 100 步）。"""
    pts = [(r["step"], ce_of(r)) for r in rows if r["step"] >= lo and ce_of(r) is not None]
    if len(pts) < 3:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom <= 0:
        return None
    return 100.0 * sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def trailing_mean(rows: List[dict], n_rows: int = 10) -> Optional[float]:
    values = [ce_of(r) for r in rows[-n_rows:] if ce_of(r) is not None]
    return sum(values) / len(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description="vlm8b 损失监控 / 平台期判定")
    parser.add_argument("--ce-file", required=True, help="run 的 per_budget_ce.jsonl")
    parser.add_argument("--ref", default="", help="参照 run 的 per_budget_ce.jsonl（如 run1）")
    parser.add_argument("--window", type=float, default=400, help="对比窗口（步）")
    parser.add_argument("--min-steps", type=float, default=1200, help="至少跑多少步才判平台")
    parser.add_argument("--rel-tol", type=float, default=0.02)
    parser.add_argument("--abs-tol", type=float, default=0.010)
    parser.add_argument("--abs-tol-loss", type=float, default=0.10,
                        help="loss 判平台的绝对阈值（loss 尺度 ~7）")
    parser.add_argument("--gn-mult", type=float, default=3.0)
    args = parser.parse_args()

    rows = load_rows(args.ce_file)
    if not rows:
        print("[monitor] 没有数据（训练可能还没到第一个 logging step）")
        return 0
    latest = rows[-1]
    step = latest["step"]
    ce_now = ce_of(latest)
    ce_now = float("nan") if ce_now is None else ce_now

    def fmt(value, spec: str) -> str:
        return fmt_na if value is None else format(value, spec)

    fmt_na = "NA"
    print(f"[monitor] file={args.ce_file}")
    print(f"[monitor] rows={len(rows)} latest_step={step} epoch={latest.get('epoch')} "
          f"ce_mean={ce_now:.4f} lr={fmt(latest.get('learning_rate'), '.3g')} "
          f"grad_norm={fmt(latest.get('grad_norm'), '.1f')}")
    ks = latest.get("budget_ks") or []
    print("[monitor] CE(k): " + " ".join(f"k{k}={v:.4f}" for k, v in zip(ks, latest[CE_KEY])))

    if args.ref:
        ref_rows = load_rows(args.ref)
        ref_ce = trailing_mean(ref_rows) if ref_rows else None
        if ref_ce is not None:
            print(f"[monitor] ref(最后10行)={ref_ce:.4f} → 当前 {ce_now:.4f} "
                  f"({(ce_now - ref_ce) / ref_ce * 100:+.1f}%)")

    first_step = rows[0]["step"]
    span = step - first_step
    # 最近 4 个窗口的均值序列：看「单调下降」还是「走平」（每行=2 张图单 batch，噪声大）
    series, hi = [], step + 1
    for _ in range(4):
        lo = hi - args.window
        mean = window_mean(rows, lo, hi)
        if mean is None:
            break
        series.append((lo, hi, mean))
        hi = lo
    if series:
        print("[monitor] 最近窗口系列: " + " | ".join(
            f"{lo:g}-{hi:g}:{mean:.4f}" for lo, hi, mean in reversed(series)))
    print(f"[monitor] window={args.window:g} steps, first_step={first_step}")
    verdict = "INSUFFICIENT"
    exit_code = 0
    if span >= args.min_steps:
        last_lo = step - args.window
        prev_lo = last_lo - args.window
        last = window_mean(rows, last_lo, step + 1)
        prev = window_mean(rows, prev_lo, last_lo)
        early = window_mean(rows, first_step + 1, first_step + args.window)
        s = slope_per_100(rows, last_lo)
        gns = [r.get("grad_norm") for r in rows if r.get("grad_norm") is not None]
        gn_last = statistics.fmean(gns[-5:]) if gns else None
        gn_ref = statistics.fmean(gns[:20]) if gns else None
        print(f"[monitor] CE窗口均值: prev({prev_lo:g}..{last_lo:g})={prev:.4f} "
              f"last({last_lo:g}..{step})={last:.4f} "
              f"Δ={last - prev:+.4f} ({(last - prev) / prev * 100:+.2f}%)")
        print(f"[monitor] 斜率(最后窗口)={s:+.5f}/100步  early={early:.4f}  "
              f"grad_norm last={gn_last:.1f} early={gn_ref:.1f}")
        loss_last = loss_window(rows, last_lo, step + 1)
        loss_prev = loss_window(rows, prev_lo, last_lo)
        if loss_last is not None and loss_prev:
            loss_rel = (loss_last - loss_prev) / loss_prev
            print(f"[monitor] loss窗口均值(主判据): prev={loss_prev:.3f} last={loss_last:.3f} "
                  f"Δ={loss_last - loss_prev:+.3f} ({loss_rel * 100:+.2f}%)")
            if loss_rel > args.rel_tol or (gn_last is not None and gn_ref and
                                           gn_last > args.gn_mult * max(gn_ref, 1e-6)):
                verdict, exit_code = "DIVERGING", 2
            elif loss_rel > -max(args.abs_tol_loss / max(loss_prev, 1e-9), args.rel_tol) and \
                    (loss_last - loss_prev) > -args.abs_tol_loss:
                verdict, exit_code = "PLATEAU", 1
            else:
                verdict = "IMPROVING"
        else:  # 没有 loss 字段时退回 CE 判据
            print("[monitor] 无 loss 字段，退回 CE 判据（噪声大，慎用）")
            if gn_last is not None and gn_ref and (
                    last > prev * (1 + args.rel_tol) or
                    gn_last > args.gn_mult * max(gn_ref, 1e-6)):
                verdict, exit_code = "DIVERGING", 2
            elif (last - prev) > -max(args.abs_tol, prev * args.rel_tol):
                verdict, exit_code = "PLATEAU", 1
            else:
                verdict = "IMPROVING"
    else:
        print(f"[monitor] 只跑了 {span:g} 步 < min-steps={args.min_steps:g}，暂不判定")

    advice = {"PLATEAU": "损失基本不降 → 建议动 lr（LM 1e-5→3e-5，从 run1/final 重启）",
              "DIVERGING": "损失/梯度变差 → 检查，必要时回滚 lr",
              "IMPROVING": "还在降 → 保持当前超参继续跑",
              "INSUFFICIENT": "数据不足 → 继续跑"}[verdict]
    print(f"[monitor] VERDICT={verdict}  →  {advice}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
