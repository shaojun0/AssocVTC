#!/usr/bin/env python3
"""回归门 — 新代码默认路径逐位不变性检查（task-3 步骤 1）
================================================================
比较"发布版代码产出的 stepwise json"(A) 与"新代码、不加任何新开关产出的
stepwise json"(B)：非数值字段必须完全相同；数值字段必须**逐位相等**。

用法::
    python doc/2026-10-16/artifacts/alit/gate_diff.py \
      --a .../stepwise_stdattn_k64_rec_published_code.json \
      --b .../stepwise_gate_stdattn_k64_rec.json \
      --out .../gate_diff_stdattn_rec.json
退出码: 0 = PASS(逐位一致) / 3 = PASS_TOLERANT(仅浮点末位差, 需 Lead 裁定) / 4 = FAIL
"""
import argparse
import json
import math
import os


def walk(a, b, path, diffs, nonnum, tol):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                diffs.append({"path": f"{path}.{k}", "kind": "missing_key",
                              "a": k in a, "b": k in b})
            else:
                walk(a[k], b[k], f"{path}.{k}", diffs, nonnum, tol)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append({"path": path, "kind": "len", "a": len(a), "b": len(b)})
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", diffs, nonnum, tol)
        return
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        if a != b:
            diffs.append({"path": path, "kind": "value", "a": a, "b": b})
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        d = abs(float(a) - float(b))
        if d != 0.0:
            diffs.append({"path": path, "kind": "num", "a": a, "b": b,
                          "abs_diff": d, "within_tol": bool(d <= tol)})
        return
    if a != b:
        nonnum.append({"path": path, "a": a, "b": b})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="发布版代码产出（基准）")
    ap.add_argument("--b", required=True, help="新代码默认开关产出")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()

    A = json.load(open(args.a))
    B = json.load(open(args.b))
    diffs, nonnum = [], []
    walk(A, B, "$", diffs, nonnum, args.tol)
    nums = [d for d in diffs if d["kind"] == "num"]
    struct = [d for d in diffs if d["kind"] != "num"]
    maxdiff = max([d["abs_diff"] for d in nums], default=0.0)
    if not diffs and not nonnum:
        verdict = "PASS"
    elif not struct and not nonnum and all(d["within_tol"] for d in nums):
        verdict = "PASS_TOLERANT"
    else:
        verdict = "FAIL"
    out = {"verdict": verdict, "a": args.a, "b": args.b,
           "n_numeric_diffs": len(nums), "n_struct_diffs": len(struct),
           "n_nonnum_diffs": len(nonnum), "max_abs_diff": maxdiff,
           "tol": args.tol, "verdict_rule": "PASS=逐位一致; PASS_TOLERANT=仅末位浮点差; FAIL=其它",
           "structural_diffs": struct[:20], "nonnum_diffs": nonnum[:20],
           "numeric_diff_sample": nums[:20]}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"[gate] verdict={verdict} numeric_diffs={len(nums)} struct={len(struct)} "
          f"nonnum={len(nonnum)} max|Δ|={maxdiff:.3e}")
    print(json.dumps({k: out[k] for k in ("verdict", "max_abs_diff",
                                          "n_numeric_diffs")}, ensure_ascii=False))
    raise SystemExit(0 if verdict == "PASS" else (3 if verdict == "PASS_TOLERANT" else 4))


if __name__ == "__main__":
    main()
