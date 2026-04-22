#!/bin/bash
# Train VAE on all training data using 6 GPUs (DDP)
# Effective batch size = 6 GPUs × 8 per-GPU = 48

torchrun --nproc_per_node=6 train.py \
    --phase vae \
    --img_h 160 --img_w 240 \
    --batch_size 8 \
    --epochs 300 \
    --lr 1e-3 \
    --beta_kl 1e-4 \
    --save_every 5 \
    --num_workers 4 \
    --checkpoint checkpoints/vae_epoch0200.pt \
    --checkpoint_dir checkpoints \
    2>&1 | tee logs/train_vae.log
