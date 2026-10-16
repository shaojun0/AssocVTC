#!/bin/bash
cd /root/AssocVTC
A=doc/2026-10-16/artifacts/alit
P=/root/autodl-tmp
for spec in "linattn/k64:$P/srdiff_linattn/k64/ckpt_best.pt" \
            "stdattn_k64:$P/srdiff_stdattn/stdattn_k64/ckpt_best.pt" \
            "linattn_k64_rec:$P/srdiff_linattn/k64_rec/ckpt_best.pt" \
            "stdattn_k64_rec:$P/srdiff_stdattn/stdattn_k64_rec/ckpt_best.pt"; do
  tag=${spec%%:*}; ck=${spec#*:}
  echo "===== $tag $(date -Is) ====="
  /root/miniconda3/bin/python eval_stepwise_linattn.py --ckpt "$ck" --num_images 0 \
    --batch_size 64 --num_workers 16 \
    --out $A/stepwise_$tag.json --dump_per_image $A/per_image_$tag.json
  echo "EXIT=$? $tag"
done
echo ALL_STEPWISE_DONE
