#!/bin/bash
# Pull a finished remote run into local artifacts/<name>/ and print the per-epoch val table.
# usage: ./collect_result.sh <remote_run_dir> <local_name>
set -u
R="${1:?remote run dir}"; N="${2:?local name}"
D=/home/linaro/dsh/tok448_work/artifacts/$N
mkdir -p "$D"
RSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p 17677 root@connect.sha1.seetacloud.com"
timeout 60 $RSH "cd $R && tar -cf - status s2.log s2/args.json s2/eval_val.json s2/train_log.jsonl s2/status.json \$(ls s2/*.json s2/*fast*.json 2>/dev/null | head -5) 2>/dev/null" | tar -xf - -C "$D" 2>/dev/null
timeout 60 scp -o StrictHostKeyChecking=no -P 17677 -r root@connect.sha1.seetacloud.com:$R/s2/samples "$D/" 2>/dev/null
echo "=== per-epoch val ($N) ==="
python3 - "$D" <<'PY'
import json,sys,os
d=sys.argv[1]
p=os.path.join(d,"s2","train_log.jsonl")
rows=[]
for line in open(p):
    r=json.loads(line)
    if r.get("split")=="val":
        rows.append(r)
print("epoch | psnr | ssim | l1_norm | gain_over_patchDC")
for r in rows:
    print(r.get("epoch"), round(r["psnr"],3), round(r["ssim"],4), round(r["l1_norm"],4),
          round(r.get("psnr_gain_over_patch_dc",float("nan")),3), sep=" | ")
print("--- eval_val.json ---")
print(open(os.path.join(d,"s2","eval_val.json")).read())
print("--- fast json files ---")
for f in sorted(os.listdir(d)):
    if f.endswith(".json") and "fast" in f:
        print(f, open(os.path.join(d,f)).read()[:1500])
PY
