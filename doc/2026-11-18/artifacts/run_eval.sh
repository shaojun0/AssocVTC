#!/bin/bash
cd /root/AssocVTC_tok448
E=/root/autodl-tmp/tok448_v2/eval
mkdir -p $E
S1=/root/autodl-tmp/tok448_v2/s1/ckpt_best.pt
S2=/root/autodl-tmp/tok448_v2/s2/ckpt_best.pt
IN=/root/autodl-tmp/datasets/imagenet100
SA=/root/autodl-tmp/savoias
run() { # gpu tag ckpt extra...
  local g=$1 tag=$2 ck=$3; shift 3
  SDPA_BACKEND=math MUSA_VISIBLE_DEVICES=$g nohup python eval_stepwise_linattn.py \
    --ckpt $ck --batch_size 64 --num_workers 8 --amp 1 \
    --out $E/summary_$tag.json --dump_per_image $E/per_image_$tag.json \
    "$@" > $E/$tag.log 2>&1 < /dev/null &
}
run 0 in100_s2 $S2 --data_dir $IN --val_per_class 50 --domain iid
run 1 savoias_s2 $S2 --data_dir_ood $SA --domain savoias
run 2 in100_s1 $S1 --data_dir $IN --val_per_class 50 --domain iid
run 3 savoias_s1 $S1 --data_dir_ood $SA --domain savoias
echo "EVAL_LAUNCHED"
