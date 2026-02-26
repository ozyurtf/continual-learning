"""
train.py — Training script for CLEVRER video prediction.

Three phases:
  --phase vae       : Train VAE (frame reconstruction + KL).
  --phase temporal  : Train temporal model (all losses). VAE is frozen by default.
  --phase joint     : Joint fine-tuning — both VAE and temporal model with LR scaling.

Usage examples
--------------
  # Phase 1 — local Mac test (1 folder, small config)
  python train.py --phase vae --video_folders video_00000-01000 \
      --img_h 128 --img_w 128 --batch_size 2 --epochs 5

  # Phase 2
  python train.py --phase temporal --video_folders video_00000-01000 \
      --img_h 128 --img_w 128 --batch_size 2 --epochs 5 \
      --vae_checkpoint checkpoints/vae_epoch05.pt

  # Resume
  python train.py --phase temporal ... --checkpoint checkpoints/temporal_epoch03.pt
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from config import get_args
from models import VAE, TemporalModel
from dataset import VideoDataset, collate_fn
from utils import (
    backward_warp, downscale_flow,
    focal_loss, save_checkpoint, load_checkpoint,
)


# ─────────────────────────────────────────────────────────────────────────────
# VAE training
# ─────────────────────────────────────────────────────────────────────────────

def train_vae(args):
    device = args.device
    print(f"[VAE] device={device}  img={args.img_h}×{args.img_w}  c_z={args.c_z}")

    dataset = VideoDataset(
        data_root=args.data_root,
        split="train",
        seq_len=1,           # single frames for VAE
        img_h=args.img_h,
        img_w=args.img_w,
        flow_dir=args.flow_dir,
        video_folders=args.video_folders,
        max_videos=args.max_videos,
        n_max=args.n_max_objects,
        c_z=args.c_z,
        h_z=args.h_z,
        w_z=args.w_z,
        random_start=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    vae = VAE(c_z=args.c_z).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)

    start_epoch = 0
    if args.checkpoint is not None:
        start_epoch = load_checkpoint(args.checkpoint, vae, optimizer, device)
        print(f"  Resumed from epoch {start_epoch}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        vae.train()
        total_loss = total_recon = total_kl = 0.0
        n_batches = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            frames = batch["frames"].to(device)  # (B, 1, 3, H, W) since seq_len=1
            frames = frames[:, 0]                # (B, 3, H, W)

            recon, mu, log_var = vae(frames)
            loss_recon = F.l1_loss(recon, frames)
            loss_kl = VAE.kl_loss(mu, log_var)
            loss = loss_recon + args.beta_kl * loss_kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss  += loss.item()
            total_recon += loss_recon.item()
            total_kl    += loss_kl.item()
            n_batches   += 1

        avg = total_loss / max(n_batches, 1)
        avg_r = total_recon / max(n_batches, 1)
        avg_k = total_kl / max(n_batches, 1)
        print(f"[VAE] epoch {epoch:4d}  loss={avg:.4f}  recon={avg_r:.4f}  kl={avg_k:.4f}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            path = os.path.join(args.checkpoint_dir, f"vae_epoch{epoch:04d}.pt")
            save_checkpoint({"model": vae.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "epoch": epoch}, path)
            print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Temporal / joint training
# ─────────────────────────────────────────────────────────────────────────────

def compute_temporal_losses(outputs, batch, vae, args, device):
    """
    Compute all temporal losses averaged across sequence steps and batch.

    Returns:
        total_loss : scalar tensor
        metrics    : dict of scalar floats for logging
    """
    T = len(outputs)  # T-1 prediction steps
    frames = batch["frames"].to(device)         # (B, T, 3, H, W)

    # Ground-truth latents: encode with frozen VAE or use precomputed
    if batch["latents"] is not None:
        z_gt = batch["latents"].to(device)       # (B, T, C_z, H_z, W_z)
    else:
        with torch.no_grad():
            B, T_all, C, H, W = frames.shape
            flat = frames.view(B * T_all, C, H, W)
            z_flat, _, _ = vae.encode(flat)
            z_gt = z_flat.view(B, T_all, *z_flat.shape[1:])

    # Precomputed supervision signals
    flows_gt       = batch["flows"].to(device)      if batch["flows"]         is not None else None
    occ_gt         = batch["occ_masks"].to(device)  if batch["occ_masks"]     is not None else None
    phys_flows_gt  = batch["physics_flows"].to(device) if batch["physics_flows"] is not None else None
    col_labels     = batch["collision_labels"].to(device)  # (B, T-1)
    obj_states_gt  = batch["obj_states"].to(device) if batch["obj_states"]    is not None else None

    losses = {}

    loss_latent = loss_flow = loss_physics = loss_occ = 0.0
    loss_warp = loss_residual = loss_state = loss_collision = 0.0

    for t, out in enumerate(outputs):
        z_next = z_gt[:, t + 1]                    # (B, C_z, H_z, W_z)
        frame_next = frames[:, t + 1]              # (B, 3, H, W)
        z_t = z_gt[:, t]

        # L_latent
        loss_latent += F.l1_loss(out["final_z"], z_next)

        # L_flow (supervised against RAFT flow in latent space)
        if flows_gt is not None:
            flow_t = flows_gt[:, t]                # (B, 2, H_z, W_z)
            loss_flow += out["flow_z"].sub(flow_t).pow(2).sum(dim=1).sqrt().mean()

        # L_physics_flow
        if phys_flows_gt is not None:
            loss_physics += F.l1_loss(out["flow_z"], phys_flows_gt[:, t])

        # L_occ
        if occ_gt is not None:
            loss_occ += F.binary_cross_entropy(out["mask"], occ_gt[:, t])

        # L_warp  (decode warped_z and compare to actual next frame)
        warped_frame = vae.decode(out["warped_z"])
        loss_warp += F.l1_loss(warped_frame, frame_next)

        # L_residual  (residual should explain what warping with GT flow misses)
        if flows_gt is not None:
            flow_t = flows_gt[:, t]
            warped_z_gt = backward_warp(z_t, flow_t)
            residual_target = z_next - warped_z_gt
            loss_residual += F.l1_loss(out["residual"], residual_target.detach())

        # L_state
        if obj_states_gt is not None:
            # Predict next-step state; ground truth at t+1
            state_target = obj_states_gt[:, t + 1]    # (B, N_max, 6)
            loss_state += F.l1_loss(out["state_pred"], state_target)

        # L_collision
        col_t = col_labels[:, t]                   # (B,)
        loss_collision += focal_loss(out["collision_logit"], col_t)

    # Average over T steps
    n = float(T)
    w = args

    total = (
        w.w_latent    * loss_latent / n
        + w.w_flow    * loss_flow / n
        + w.w_physics_flow * loss_physics / n
        + w.w_occ     * loss_occ / n
        + w.w_warp    * loss_warp / n
        + w.w_residual * loss_residual / n
        + w.w_state   * loss_state / n
        + w.w_collision * loss_collision / n
    )

    metrics = {
        "latent":    (loss_latent / n).item() if torch.is_tensor(loss_latent) else loss_latent / n,
        "flow":      (loss_flow / n).item()   if torch.is_tensor(loss_flow)   else 0.0,
        "occ":       (loss_occ / n).item()    if torch.is_tensor(loss_occ)    else 0.0,
        "warp":      (loss_warp / n).item(),
        "residual":  (loss_residual / n).item() if torch.is_tensor(loss_residual) else 0.0,
        "state":     (loss_state / n).item()  if torch.is_tensor(loss_state)  else 0.0,
        "collision": (loss_collision / n).item(),
        "total":     total.item(),
    }

    return total, metrics


def train_temporal(args, joint: bool = False):
    device = args.device
    phase_name = "Joint" if joint else "Temporal"
    print(f"[{phase_name}] device={device}  seq_len={args.seq_len}")

    # ── Load VAE ──────────────────────────────────────────────────────────────
    if args.vae_checkpoint is None:
        print("ERROR: --vae_checkpoint is required for temporal/joint training.")
        sys.exit(1)

    vae = VAE(c_z=args.c_z).to(device)
    load_checkpoint(args.vae_checkpoint, vae, device=device)

    if joint or args.unfreeze_vae:
        vae.train()
    else:
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)

    # ── Temporal model ────────────────────────────────────────────────────────
    temporal = TemporalModel(
        c_z=args.c_z, c_lstm=args.c_lstm,
        max_disp=args.max_disp, n_max=args.n_max_objects
    ).to(device)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    if joint or args.unfreeze_vae:
        param_groups = [
            {"params": temporal.parameters(), "lr": args.lr},
            {"params": vae.parameters(),      "lr": args.lr * args.vae_lr_scale},
        ]
    else:
        param_groups = [{"params": temporal.parameters(), "lr": args.lr}]

    optimizer = torch.optim.Adam(param_groups)

    start_epoch = 0
    if args.checkpoint is not None:
        # Load temporal checkpoint
        ckpt = torch.load(args.checkpoint, map_location=device)
        temporal.load_state_dict(ckpt["temporal"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        if "vae" in ckpt:
            vae.load_state_dict(ckpt["vae"])
        print(f"  Resumed from epoch {start_epoch}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = VideoDataset(
        data_root=args.data_root,
        split="train",
        seq_len=args.seq_len,
        img_h=args.img_h,
        img_w=args.img_w,
        flow_dir=args.flow_dir,
        video_folders=args.video_folders,
        max_videos=args.max_videos,
        n_max=args.n_max_objects,
        c_z=args.c_z,
        h_z=args.h_z,
        w_z=args.w_z,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    prefix = "joint" if joint else "temporal"

    for epoch in range(start_epoch + 1, args.epochs + 1):
        temporal.train()
        if joint or args.unfreeze_vae:
            vae.train()

        total_loss = 0.0
        n_batches = 0
        epoch_metrics = {}

        for batch in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            frames = batch["frames"].to(device)   # (B, T, 3, H, W)
            B, T, C, H, W = frames.shape

            # Encode frames to latents (use precomputed if available)
            if batch["latents"] is not None:
                z_seq = batch["latents"].to(device)
            else:
                with torch.set_grad_enabled(joint or args.unfreeze_vae):
                    flat = frames.view(B * T, C, H, W)
                    z_flat, _, _ = vae.encode(flat)
                z_seq = z_flat.view(B, T, *z_flat.shape[1:])

            outputs, _ = temporal(z_seq, vae_decoder=vae.decoder)

            loss, metrics = compute_temporal_losses(outputs, batch, vae, args, device)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(temporal.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1
            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v

        avg_total = total_loss / max(n_batches, 1)
        parts = "  ".join(f"{k}={v / max(n_batches, 1):.4f}"
                          for k, v in epoch_metrics.items() if k != "total")
        print(f"[{phase_name}] epoch {epoch:4d}  total={avg_total:.4f}  {parts}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_data = {
                "temporal":  temporal.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch":     epoch,
            }
            if joint or args.unfreeze_vae:
                ckpt_data["vae"] = vae.state_dict()
            path = os.path.join(args.checkpoint_dir,
                                f"{prefix}_epoch{epoch:04d}.pt")
            save_checkpoint(ckpt_data, path)
            print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    if args.phase == "vae":
        train_vae(args)
    elif args.phase == "temporal":
        train_temporal(args, joint=False)
    elif args.phase == "joint":
        train_temporal(args, joint=True)
    else:
        print(f"Unknown phase: {args.phase}")
        sys.exit(1)


if __name__ == "__main__":
    main()
