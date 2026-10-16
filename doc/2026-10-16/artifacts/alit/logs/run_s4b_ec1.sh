#!/bin/bash
cd /root/AssocVTC
A=doc/2026-10-16/artifacts/alit
R=/root/autodl-tmp/srdiff_stdattn/stdattn_k64_rec/ckpt_best.pt
C=/root/autodl-tmp/alit_cache/zs_stdattn_k64_rec
mkdir -p /root/autodl-tmp/alit_cache
BASE="--ckpt $R --data_dir /root/autodl-tmp/datasets/imagenet100 --val_per_class 50 --val_n 0 \
 --train_per_class 100 --tsc_json $A/per_image_stdattn_k64_rec.json --tsc_taus auto --tsc_n_taus 9 \
 --ks 1,4,8,12,16,24,32,40,48,56,64 --batch_size 64 --num_workers 16 --amp 1 \
 --cache $C --n_boot 1000 --seed 0"
echo "===== EC1 primary $(date -Is) ====="
/root/miniconda3/bin/python -u tools/alit_task_fidelity.py $BASE --out $A/task_fidelity_stdattn_rec.json
echo "EC1_PRIMARY_RC=$?"
echo "===== EC1 sensitivity: use_cls=1 ====="
/root/miniconda3/bin/python -u tools/alit_task_fidelity.py $BASE --use_cls 1 --out $A/task_fidelity_stdattn_rec_cls.json
echo "EC1_CLS_RC=$?"
echo "===== EC1 sensitivity: probe=lstsq ====="
/root/miniconda3/bin/python -u tools/alit_task_fidelity.py $BASE --probe lstsq --out $A/task_fidelity_stdattn_rec_lstsq.json
echo "EC1_LSTSQ_RC=$?"
echo EC1_CHAIN_DONE
