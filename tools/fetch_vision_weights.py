"""从 HF safetensors 分片里**只取视觉编码器权重**（按 tensor byte-range 下载），
避免为了一个 ~0.3–0.8B 的 vision tower 去下 5–13GB 的整片（这些模型的 LLM 主体是
300–500GB）。

做法：
  1. Range 读 safetensors 头（前 8 字节 = header 长度，header 是 JSON）；
  2. header 里每个 tensor 有 data_offsets [start,end) ⇒ 只取 `--prefix` 匹配的键；
  3. 把相邻区间合并（--gap_mb，默认 8MB）后逐个 Range 下载；
  4. 还原成 numpy → 存成紧凑 safetensors（键保持原名，加载端自己 strip 前缀）。

若服务器不支持 Range（返回 200 而不是 206）⇒ 下载整片再抽取（会慢/占盘，但能兜底）。

用法：
  python3 fetch_vision_weights.py --repo zai-org/GLM-5.3-Flash \
      --shard model-00062-of-00062.safetensors --prefix model.visual. \
      --out /root/autodl-tmp/vis_enc/glm_visual.safetensors
"""
import argparse
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request

UA = {"User-Agent": "Mozilla/5.0"}


def _open(url, start=None, end=None, timeout=300):
    req = urllib.request.Request(url, headers=dict(UA))
    if start is not None:
        rng = f"bytes={start}-{'' if end is None else end}"
        req.add_header("Range", rng)
    return urllib.request.urlopen(req, timeout=timeout)


def _read(url, start=None, end=None, timeout=300):
    with _open(url, start, end, timeout) as r:
        return r.read(), r.status, dict(r.headers)


def read_header(url):
    first, st, hd = _read(url, 0, 7)
    assert st == 206, f"服务器不支持 Range（HTTP {st}）: {url}"
    n = struct.unpack("<Q", first)[0]
    hdr, st2, _ = _read(url, 8, 8 + n - 1)
    assert st2 == 206 and len(hdr) == n
    return json.loads(hdr), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--endpoint", default="https://hf-mirror.com")
    ap.add_argument("--gap_mb", type=float, default=8.0)
    ap.add_argument("--revision", default="main")
    args = ap.parse_args()

    url = f"{args.endpoint}/{args.repo}/resolve/{args.revision}/{args.shard}"
    t0 = time.time()
    hdr, hlen = read_header(url)
    ents = []
    for k, v in hdr.items():
        if k == "__metadata__" or not k.startswith(args.prefix):
            continue
        s, e = v["data_offsets"]
        ents.append((int(s), int(e), k, v["dtype"], v["shape"]))
    if not ents:
        print(f"[fetch] WARN no keys with prefix {args.prefix!r} in {args.shard}")
        return
    ents.sort()
    gap = args.gap_mb * 1024 * 1024
    groups, cur = [], []
    for e in ents:
        if cur and e[0] - cur[-1][1] > gap:
            groups.append(cur); cur = []
        cur.append(e)
    if cur:
        groups.append(cur)
    total_bytes = sum(e[1] - e[0] for e in ents)
    span_bytes = sum(g[-1][1] - g[0][0] for g in groups)
    print(f"[fetch] {args.repo} {args.shard}: {len(ents)} tensors, "
          f"payload={total_bytes/1e9:.2f}GB, span={span_bytes/1e9:.2f}GB, "
          f"groups={len(groups)} header={hlen/1e6:.1f}MB", flush=True)

    import numpy as np
    from safetensors.numpy import save_file

    DT = {"F64": np.float64, "F32": np.float32, "F16": np.float16,
          "I64": np.int64, "I32": np.int32, "I16": np.int16,
          "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_}
    out = {}
    got = 0
    for gi, g in enumerate(groups):
        lo, hi = g[0][0], g[-1][1]
        data, st, _ = _read(url, lo, hi - 1)
        assert st == 206, f"range 请求返回 {st}"
        assert len(data) == hi - lo, (len(data), hi - lo)
        for (s, e, k, dt, shape) in g:
            raw = data[s - lo:e - lo]
            if dt == "BF16":
                u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
                arr = u.view(np.float32).reshape(shape)
            else:
                arr = np.frombuffer(raw, dtype=DT[dt]).reshape(shape)
            out[k] = np.ascontiguousarray(arr)
        got += len(g)
        print(f"[fetch]   group {gi+1}/{len(groups)} {((hi-lo)/1e6):.1f}MB "
              f"keys={len(g)} ({got}/{len(ents)}) {time.time()-t0:.0f}s", flush=True)
    d = os.path.dirname(args.out)
    if d:
        os.makedirs(d, exist_ok=True)
    save_file(out, args.out)
    sz = os.path.getsize(args.out) / 1e9
    print(f"[fetch] -> {args.out} {sz:.2f}GB keys={len(out)} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
