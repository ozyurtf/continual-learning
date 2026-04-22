#!/bin/bash
# Train temporal model using 6 GPUs (DDP). VAE is frozen.
# Effective batch size = 6 GPUs × 4 per-GPU = 24

torchrun --nproc_per_node=6 train.py \
    --phase temporal \
    --img_h 160 --img_w 240 \
    --batch_size 8 \
    --seq_len 10 \
    --epochs 200 \
    --lr 1e-3 \
    --use_uncertainty_weights \
    --w_physics_flow 0.0 \
    --w_state 0.0 \
    --w_collision 0.0 \
    --save_every 5 \
    --num_workers 4 \
    --checkpoint_dir checkpoints \
    --vae_checkpoint checkpoints/vae_epoch0200.pt \
    2>&1 | tee logs/train_temporal.log