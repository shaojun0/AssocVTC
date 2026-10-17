#!/bin/bash
# ALIT stage-1 (latent distillation) 单卡 4080 32G 复现：alit_tiny(=论文 Ours-S, 8层x512, 52.8M 可训)
# bs16 x accum16 = 有效 batch 256（对齐论文 8卡x32），blr 1.5e-4，30 epoch 余弦退火
set -u
cd /root/autodl-tmp/alit
export PYTHONPATH=/root/autodl-tmp/alit
export WANDB_MODE=offline
OUT=/root/autodl-tmp/alit_out/s1_tiny_in100
EPOCHS=${EPOCHS:-30}
mkdir -p "$OUT"
while :; do
  RESUME=""
  LAST=$OUT/checkpoint-last.pth
  if [ -f "$LAST" ]; then
    CUR=$(/root/miniconda3/bin/python -c "import torch;print(torch.load('$LAST',map_location='cpu',weights_only=False).get('epoch',-1))" 2>/dev/null || echo -1)
    echo "[driver] found last ckpt epoch=$CUR"
    if [ "$CUR" -ge $((EPOCHS-1)) ]; then echo "[driver] training finished"; break; fi
    RESUME="--resume $LAST"
  fi
  echo "[driver] launch $(date) resume=${RESUME:-none}"
  /root/miniconda3/bin/torchrun --nproc_per_node=1 --master_port=12347 main_pretrain.py \
    --batch_size 16 --accum_iter 16 --num_workers 12 --model alit_tiny \
    --base_tokenizer vqgan --quantize_latent --factorize_latent \
    --epochs $EPOCHS --warmup_epochs 3 --blr 1.5e-4 --weight_decay 0.05 \
    --save_freq 5 --output_dir "$OUT" --data_path /root/autodl-tmp/datasets/alit_in100 \
    $RESUME >> "$OUT/train.log" 2>&1
  RC=$?
  echo "[driver] exit rc=$RC at $(date)"
  sleep 20
done
