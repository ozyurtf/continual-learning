"""
train.py — Training script for CLEVRER video prediction.

Three phases:
  --phase vae       : Train VAE (frame reconstruction + KL).
  --phase temporal  : Train temporal model (all losses). VAE is frozen by default.
  --phase joint     : Joint fine-tuning — both VAE and temporal model with LR scaling.

Single-GPU usage
----------------
  python train.py --phase vae --img_h 160 --img_w 240 --batch_size 8 --epochs 50

Multi-GPU usage (DDP via torchrun)
-----------------------------------
  torchrun --nproc_per_node=8 train.py --phase vae \
      --img_h 160 --img_w 240 --batch_size 8 --epochs 50

  torchrun --nproc_per_node=8 train.py --phase temporal \
      --img_h 160 --img_w 240 --batch_size 4 --epochs 50 \
      --vae_checkpoint checkpoints/vae_epoch0050.pt

Notes
-----
  - batch_size is *per GPU*. Effective batch = batch_size × nproc_per_node.
  - Checkpoints and prints are only written by rank 0.
  - DDP is only active when WORLD_SIZE > 1 (torchrun sets this env var).
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from config import get_args
from models import VAE, TemporalModel
from dataset import VideoDataset, collate_fn
from utils import (
    backward_warp, downscale_flow,
    focal_loss, perceptual_loss, save_checkpoint, load_checkpoint,
)


def is_ddp() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def get_rank() -> int:
    return dist.get_rank() if is_ddp() else 0


def is_main() -> bool:
    return get_rank() == 0


def init_ddp() -> torch.device:
    """
    Initialize the process group if running under torchrun.
    Returns the device assigned to this process.
    Falls back to args.device if not under torchrun.
    """
    rank = int(os.environ.get("RANK", -1))
    if rank == -1:
        # Not launched with torchrun — single process
        return None

    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")


def cleanup_ddp():
    if is_ddp():
        dist.destroy_process_group()


def unwrap(model):
    """Return the underlying module when wrapped in DDP."""
    return model.module if isinstance(model, DDP) else model


def train_vae(args, device):
    if is_main():
        print(f"[VAE] device={device}  img={args.img_h}×{args.img_w}  c_z={args.c_z}  "
              f"world_size={dist.get_world_size() if is_ddp() else 1}")

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

    if is_ddp():
        sampler = DistributedSampler(dataset, shuffle=True)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    else:
        sampler = None
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
        if is_ddp():
            dist.barrier()  # ensure rank 0 reads before others
        start_epoch = load_checkpoint(args.checkpoint, vae, optimizer, device)
        if is_main():
            print(f"  Resumed from epoch {start_epoch}")

    if is_ddp():
        vae = DDP(vae, device_ids=[device.index])

    if is_main():
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)

        unwrap(vae).train()
        total_loss = total_recon = total_kl = total_perc = 0.0
        n_batches = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}",
                          leave=False, disable=not is_main()):
            frames = batch["frames"].to(device)  # (B, 1, 3, H, W) since seq_len=1
            frames = frames[:, 0]                # (B, 3, H, W)

            recon, mu, log_var = vae(frames)
            loss_recon = F.l1_loss(recon, frames)
            loss_kl    = VAE.kl_loss(mu, log_var)
            loss = loss_recon + args.beta_kl * loss_kl
            if args.w_perceptual > 0:
                loss_perc = perceptual_loss(recon, frames)
                loss = loss + args.w_perceptual * loss_perc
            else:
                loss_perc = torch.tensor(0.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss  += loss.item()
            total_recon += loss_recon.item()
            total_kl    += loss_kl.item()
            total_perc  += loss_perc.item()
            n_batches   += 1

        if is_main():
            avg   = total_loss  / max(n_batches, 1)
            avg_r = total_recon / max(n_batches, 1)
            avg_k = total_kl    / max(n_batches, 1)
            avg_p = total_perc  / max(n_batches, 1)
            print(f"[VAE] epoch {epoch:4d}  loss={avg:.4f}  recon={avg_r:.4f}  kl={avg_k:.4f}  perc={avg_p:.4f}")

            if epoch % args.save_every == 0 or epoch == args.epochs:
                path = os.path.join(args.checkpoint_dir, f"vae_epoch{epoch:04d}.pt")
                save_checkpoint({"model":     unwrap(vae).state_dict(),
                                 "optimizer": optimizer.state_dict(),
                                 "epoch":     epoch}, path)
                print(f"  Saved {path}")


# Names of the 5 core losses managed by uncertainty weighting
UNCERTAINTY_LOSS_NAMES = ["latent", "flow", "occ", "warp", "residual"]


class UncertaintyWeights(torch.nn.Module):
    """
    Learnable per-loss uncertainty weights.

    Each loss i is weighted as:
        (1 / σ_i²) * loss_i  +  log(σ_i)
      = exp(-log_var_i) * loss_i  +  0.5 * log_var_i

    log_var_i = log(σ_i²) is a learnable parameter initialized to 0 (σ²=1).
    """
    def __init__(self, n: int = 5):
        super().__init__()
        self.log_vars = torch.nn.Parameter(torch.zeros(n))

    def forward(self, losses: list) -> torch.Tensor:
        total = sum(
            torch.exp(-self.log_vars[i]) * loss + 0.5 * self.log_vars[i]
            for i, loss in enumerate(losses)
        )
        return total

    def weights_dict(self) -> dict:
        """Return current effective weights (1/σ²) for logging."""
        return {
            name: torch.exp(-self.log_vars[i]).item()
            for i, name in enumerate(UNCERTAINTY_LOSS_NAMES)
        }


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
            z_flat, _, _ = unwrap(vae).encode(flat)
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
        warped_frame = unwrap(vae).decode(out["warped_z"])
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

    # Convert floats to tensors for uniform handling
    def _t(x):
        return x / n if torch.is_tensor(x) else torch.tensor(0.0, device=device)

    core_losses = [
        _t(loss_latent),
        _t(loss_flow),
        _t(loss_occ),
        _t(loss_warp),
        _t(loss_residual),
    ]

    # Minor losses always use manual weights (typically 0)
    minor_total = (
        w.w_physics_flow * _t(loss_physics)
        + w.w_state      * _t(loss_state)
        + w.w_collision  * _t(loss_collision)
    )

    metrics = {
        "latent":    core_losses[0].item(),
        "flow":      core_losses[1].item(),
        "occ":       core_losses[2].item(),
        "warp":      core_losses[3].item(),
        "residual":  core_losses[4].item(),
        "state":     _t(loss_state).item(),
        "collision": _t(loss_collision).item(),
    }

    return core_losses, minor_total, metrics


def train_temporal(args, device, joint: bool = False):
    phase_name = "Joint" if joint else "Temporal"
    if is_main():
        print(f"[{phase_name}] device={device}  seq_len={args.seq_len}  "
              f"world_size={dist.get_world_size() if is_ddp() else 1}")

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

    temporal = TemporalModel(
        c_z=args.c_z, c_lstm=args.c_lstm,
        max_disp=args.max_disp, n_max=args.n_max_objects
    ).to(device)

    unc_weights = UncertaintyWeights(n=len(UNCERTAINTY_LOSS_NAMES)).to(device) \
                  if args.use_uncertainty_weights else None

    if joint or args.unfreeze_vae:
        param_groups = [
            {"params": temporal.parameters(), "lr": args.lr},
            {"params": vae.parameters(),      "lr": args.lr * args.vae_lr_scale},
        ]
    else:
        param_groups = [{"params": temporal.parameters(), "lr": args.lr}]

    if unc_weights is not None:
        param_groups.append({"params": unc_weights.parameters(), "lr": args.lr})

    optimizer = torch.optim.Adam(param_groups)

    start_epoch = 0
    if args.checkpoint is not None:
        if is_ddp():
            dist.barrier()
        ckpt = torch.load(args.checkpoint, map_location=device)
        temporal.load_state_dict(ckpt["temporal"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        if "vae" in ckpt:
            vae.load_state_dict(ckpt["vae"])
        if unc_weights is not None and "unc_weights" in ckpt:
            unc_weights.load_state_dict(ckpt["unc_weights"])
        if is_main():
            print(f"  Resumed from epoch {start_epoch}")

    if is_ddp():
        temporal = DDP(temporal, device_ids=[device.index])
        if joint or args.unfreeze_vae:
            vae = DDP(vae, device_ids=[device.index])

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

    if is_ddp():
        sampler = DistributedSampler(dataset, shuffle=True)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    else:
        sampler = None
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=(device.type == "cuda"),
        )

    if is_main():
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    prefix = "joint" if joint else "temporal"

    for epoch in range(start_epoch + 1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)

        unwrap(temporal).train()
        if joint or args.unfreeze_vae:
            unwrap(vae).train()

        total_loss = 0.0
        n_batches = 0
        epoch_metrics = {}

        for batch in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}",
                          leave=False, disable=not is_main()):
            frames = batch["frames"].to(device)   # (B, T, 3, H, W)
            B, T, C, H, W = frames.shape

            # Encode frames to latents (use precomputed if available)
            if batch["latents"] is not None:
                z_seq = batch["latents"].to(device)
            else:
                with torch.set_grad_enabled(joint or args.unfreeze_vae):
                    flat = frames.view(B * T, C, H, W)
                    z_flat, _, _ = unwrap(vae).encode(flat)
                z_seq = z_flat.view(B, T, *z_flat.shape[1:])

            outputs, _ = unwrap(temporal)(z_seq, vae_decoder=unwrap(vae).decoder)

            core_losses, minor_total, metrics = compute_temporal_losses(
                outputs, batch, vae, args, device
            )

            # Apply uncertainty weighting or manual weights
            if unc_weights is not None:
                loss = unc_weights(core_losses) + minor_total
            else:
                w = args
                loss = (
                    w.w_latent   * core_losses[0]
                    + w.w_flow   * core_losses[1]
                    + w.w_occ    * core_losses[2]
                    + w.w_warp   * core_losses[3]
                    + w.w_residual * core_losses[4]
                    + minor_total
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unwrap(temporal).parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1
            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v

        if is_main():
            avg_total = total_loss / max(n_batches, 1)
            parts = "  ".join(f"{k}={v / max(n_batches, 1):.4f}"
                              for k, v in epoch_metrics.items())
            print(f"[{phase_name}] epoch {epoch:4d}  total={avg_total:.4f}  {parts}")

            # Log learned uncertainty weights if active
            if unc_weights is not None:
                w_str = "  ".join(f"{k}={v:.3f}"
                                  for k, v in unc_weights.weights_dict().items())
                print(f"  [unc_weights] {w_str}")

            if epoch % args.save_every == 0 or epoch == args.epochs:
                ckpt_data = {
                    "temporal":  unwrap(temporal).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch":     epoch,
                }
                if joint or args.unfreeze_vae:
                    ckpt_data["vae"] = unwrap(vae).state_dict()
                if unc_weights is not None:
                    ckpt_data["unc_weights"] = unc_weights.state_dict()
                path = os.path.join(args.checkpoint_dir,
                                    f"{prefix}_epoch{epoch:04d}.pt")
                save_checkpoint(ckpt_data, path)
                print(f"  Saved {path}")



def main():
    args = get_args()

    # Initialise DDP (no-op if not launched with torchrun)
    ddp_device = init_ddp()
    device = ddp_device if ddp_device is not None else args.device

    if args.phase == "vae":
        train_vae(args, device)
    elif args.phase == "temporal":
        train_temporal(args, device, joint=False)
    elif args.phase == "joint":
        train_temporal(args, device, joint=True)
    else:
        print(f"Unknown phase: {args.phase}")
        sys.exit(1)

    cleanup_ddp()


if __name__ == "__main__":
    main()
