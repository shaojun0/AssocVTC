#!/bin/bash
cd /root/AssocVTC
A=doc/2026-10-16/artifacts/alit
P=/root/autodl-tmp
echo "########## STAGE1 stepwise 4-arm (published code) $(date -Is) ##########"
for spec in "linattn_k64:$P/srdiff_linattn/k64/ckpt_best.pt" \
            "stdattn_k64:$P/srdiff_stdattn/stdattn_k64/ckpt_best.pt" \
            "linattn_k64_rec:$P/srdiff_linattn/k64_rec/ckpt_best.pt" \
            "stdattn_k64_rec:$P/srdiff_stdattn/stdattn_k64_rec/ckpt_best.pt"; do
  tag=${spec%%:*}; ck=${spec#*:}
  echo "===== stepwise $tag $(date -Is) ====="
  /root/miniconda3/bin/python eval_stepwise_linattn.py --ckpt "$ck" --num_images 0 \
    --batch_size 64 --num_workers 16 --out $A/stepwise_$tag.json
  echo "EXIT=$? $tag"
done
echo "STAGE1_DONE $(date -Is)"

echo "########## STAGE2 baseline arm b1_mean $(date -Is) ##########"
mkdir -p $P/srdiff_alit/b1_mean
cd /root/AssocVTC
nohup setsid /root/miniconda3/bin/python -u train_linattn.py \
  --data_dir /root/autodl-tmp/datasets/imagenet100 \
  --output_dir /root/autodl-tmp/srdiff_alit/b1_mean \
  --enc_attn std --num_specials 64 --batch_size 64 --epochs 6 --lr 2e-4 \
  --dec_recurrent 1 --dec_grad_ckpt 1 \
  --init_from /root/autodl-tmp/srdiff_stdattn/stdattn_k64/ckpt_best.pt \
  --time_budget_h 3.0 \
  > /root/autodl-tmp/srdiff_alit/b1_mean_launch.log 2>&1 &
TPID=$!
echo $TPID > /root/autodl-tmp/srdiff_alit/b1_mean_train.pid
echo "TRAIN_PID=$TPID $(date -Is)"
sleep 20
echo "--- train log head ---"; head -30 /root/autodl-tmp/srdiff_alit/b1_mean_launch.log
