#!/bin/bash
# 下载 SAVOIAS 的 3 个子集（People-Art / Places / MSCOCO-Objects）+ complexity ground truth
# -----------------------------------------------------------------------------
# ALIT (arXiv:2411.02393) Fig.4/11/12 的人评复杂度分析用的就是这三个子集：
#   Images/Art      = People-Art, 420 张, human complexity 0-100
#   Images/Scenes   = Places,    200 张
#   Images/Objects  = MSCOCO,    200 张
# 数据源：https://github.com/esaraee/Savoias-Dataset （Saraee et al., 2018）
#
# 用法：
#   bash tools/fetch_savoias.sh [DATA_ROOT]     # 默认 /root/autodl-tmp/savoias
#
# 产物布局（供 eval_stepwise_linattn.py --data_dir_ood 递归扫描）：
#   $DATA_ROOT/{Art,Scenes,Objects}/*.jpg
#   $DATA_ROOT/gt/gt_{art,scenes,objects}.npy
# -----------------------------------------------------------------------------
set -euo pipefail

DATA_ROOT=${1:-/root/autodl-tmp/savoias}
BASE="https://raw.githubusercontent.com/esaraee/Savoias-Dataset/master"
mkdir -p "$DATA_ROOT"/{Art,Scenes,Objects,gt}

# 1) ground truth（小文件，先下）
for c in art scenes objects; do
  [ -s "$DATA_ROOT/gt/gt_$c.npy" ] && continue
  curl -sL --retry 3 --max-time 120 \
    -o "$DATA_ROOT/gt/gt_$c.npy" \
    "$BASE/Ground%20truth/npy/global_ranking_$c.npy"
done
echo "[gt] $(ls -lh "$DATA_ROOT"/gt | tail -n +2)"

# 2) 图像：并行下载 + 断点续传（已存在且非空则跳过）
gen_urls() {
  python3 - <<'PY'
import json, urllib.request
api = ("https://api.github.com/repos/esaraee/Savoias-Dataset/"
       "git/trees/master?recursive=1")
tree = json.load(urllib.request.urlopen(api, timeout=60))["tree"]
for t in tree:
    p = t["path"]
    if t["type"] == "blob" and any(
            p.startswith("Images/" + c + "/") for c in ["Art", "Scenes", "Objects"]):
        print("https://raw.githubusercontent.com/esaraee/Savoias-Dataset/master/" + p)
PY
}

gen_urls > "$DATA_ROOT/urls.txt"
wc -l "$DATA_ROOT/urls.txt"
cat "$DATA_ROOT/urls.txt" | xargs -P 32 -I{} bash -c '
  url="$1"; rel="${url#*master/}"; out="'"$DATA_ROOT"'/${rel#Images/}"
  mkdir -p "$(dirname "$out")"
  [ -s "$out" ] && exit 0
  curl -sL --retry 3 --max-time 180 -o "$out" "$url" || echo "FAIL $url"
' _ {}
echo "[images] $(for c in Art Scenes Objects; do echo -n "$c=$(ls "$DATA_ROOT/$c" | wc -l) "; done)"
echo "done: $DATA_ROOT"
