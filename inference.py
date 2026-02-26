"""
inference.py — Autoregressive video prediction with optional Test-Time Training (TTT).

For each video:
  1. Feed first `num_input_frames` real frames through the ConvLSTM to warm up hidden state.
  2. Predict the next `num_pred_frames` frames autoregressively:
       flow → warp latent → residual → decode
  3. (Optional TTT) Before step 2, fine-tune FlowHead + OcclusionHead + ResidualHead
     jointly on the input frames using a self-supervised warp-consistency loss.
     Reset weights after each video.
  4. Compute PSNR, SSIM, LPIPS on predicted vs ground-truth frames.
  5. Save predicted frames as MP4.

Usage
-----
  python inference.py \\
      --vae_checkpoint     checkpoints/vae_epoch0050.pt \\
      --temporal_checkpoint checkpoints/temporal_epoch0050.pt \\
      --video_folders      video_00000-01000 \\
      --img_h 128 --img_w 128 \\
      --num_input_frames 5 --num_pred_frames 5 \\
      --ttt --ttt_steps 10 --output_dir outputs
"""

import os
import sys
import copy
import json
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.io as tv_io
from tqdm import tqdm

from config import get_args
from models import VAE, TemporalModel
from dataset import VideoDataset, collate_fn
from utils import (
    compute_psnr, compute_ssim, compute_lpips,
    save_checkpoint, load_checkpoint,
)
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_uint8_nhwc(frames: torch.Tensor) -> np.ndarray:
    """(T, 3, H, W) float [0,1] → (T, H, W, 3) uint8 for video saving."""
    return (frames.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()


def save_mp4(frames_uint8: np.ndarray, path: str, fps: int = 24) -> None:
    """Save (T, H, W, 3) uint8 numpy array as MP4."""
    import cv2
    T, H, W, C = frames_uint8.shape
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (W, H))
    for t in range(T):
        frame_bgr = cv2.cvtColor(frames_uint8[t], cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    writer.release()


# ─────────────────────────────────────────────────────────────────────────────
# TTT
# ─────────────────────────────────────────────────────────────────────────────

def run_ttt(temporal: TemporalModel, vae: VAE,
            z_inputs: torch.Tensor, frames_inputs: torch.Tensor,
            ttt_steps: int, ttt_lr: float, device: torch.device) -> TemporalModel:
    """
    Test-Time Training: fine-tune FlowHead + OcclusionHead + ResidualHead
    on the input frames using a self-supervised warp-consistency loss.

        L_ttt = L1( decode( warp(z_t, flow_z_t) ), frame_{t+1} )

    Args:
        temporal       : TemporalModel (will be fine-tuned in-place on a *copy*)
        vae            : frozen VAE
        z_inputs       : (1, K, C_z, H_z, W_z) latents of input frames
        frames_inputs  : (1, K, 3, H, W) pixel frames (targets)
        ttt_steps      : gradient steps
        ttt_lr         : learning rate

    Returns:
        Fine-tuned temporal model (original is not modified; caller must replace).
    """
    temporal_ttt = copy.deepcopy(temporal)
    temporal_ttt.train()

    # Only fine-tune the three spatial heads
    ttt_params = (
        list(temporal_ttt.flow_head.parameters())
        + list(temporal_ttt.occ_head.parameters())
        + list(temporal_ttt.residual_head.parameters())
    )
    optimizer = torch.optim.Adam(ttt_params, lr=ttt_lr)

    K = z_inputs.shape[1]  # number of input frames

    for _ in range(ttt_steps):
        state = temporal_ttt.init_state(1, z_inputs.shape[3],
                                        z_inputs.shape[4], device)
        loss_ttt = torch.tensor(0.0, device=device)

        for t in range(K - 1):
            out = temporal_ttt.step(z_inputs[:, t], state)
            state = out["new_state"]

            # Warp frame_t using predicted flow → compare to frame_{t+1}
            warped_frame = vae.decode(out["warped_z"])
            frame_next = frames_inputs[:, t + 1]
            loss_ttt = loss_ttt + F.l1_loss(warped_frame, frame_next)

        loss_ttt = loss_ttt / max(K - 1, 1)
        optimizer.zero_grad()
        loss_ttt.backward()
        optimizer.step()

    return temporal_ttt


# ─────────────────────────────────────────────────────────────────────────────
# Single-video inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_video(vae: VAE, temporal: TemporalModel,
                  frames: torch.Tensor,    # (T_full, 3, H, W)
                  z_all: torch.Tensor,     # (T_full, C_z, H_z, W_z)
                  num_input: int,
                  num_pred: int,
                  device: torch.device):
    """
    Autoregressive prediction.

    Warm-up: run ConvLSTM on first `num_input` real latents to build hidden state.
    Predict: generate `num_pred` future latents autoregressively.

    Returns:
        pred_frames : (num_pred, 3, H, W) float32 [0,1]
    """
    B = 1  # single video
    H_z, W_z = z_all.shape[-2], z_all.shape[-1]

    state = temporal.init_state(B, H_z, W_z, device)

    # ── Warm-up ────────────────────────────────────────────────────────────
    for t in range(num_input):
        z_t = z_all[t: t + 1]  # (1, C_z, H_z, W_z)
        out = temporal.step(z_t, state)
        state = out["new_state"]

    # ── Autoregressive prediction ──────────────────────────────────────────
    pred_frames = []
    z_prev = z_all[num_input - 1: num_input]  # last real latent

    for _ in range(num_pred):
        out = temporal.step(z_prev, state)
        state = out["new_state"]

        pred_frame = vae.decode(out["final_z"])  # (1, 3, H, W)
        pred_frames.append(pred_frame[0])

        # Encode the predicted frame for the next step
        with torch.no_grad():
            z_prev, _, _ = vae.encode(pred_frame)

    return torch.stack(pred_frames, dim=0)  # (num_pred, 3, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    device = args.device

    if args.vae_checkpoint is None or args.temporal_checkpoint is None:
        print("ERROR: --vae_checkpoint and --temporal_checkpoint are required.")
        sys.exit(1)

    # ── Load models ───────────────────────────────────────────────────────────
    vae = VAE(c_z=args.c_z).to(device)
    load_checkpoint(args.vae_checkpoint, vae, device=device)
    vae.eval()

    temporal = TemporalModel(
        c_z=args.c_z, c_lstm=args.c_lstm,
        max_disp=args.max_disp, n_max=args.n_max_objects
    ).to(device)
    load_checkpoint(args.temporal_checkpoint, temporal, device=device)
    temporal.eval()

    # ── Dataset ───────────────────────────────────────────────────────────────
    T_needed = args.num_input_frames + args.num_pred_frames
    dataset = VideoDataset(
        data_root=args.data_root,
        split="val",
        seq_len=T_needed,
        img_h=args.img_h,
        img_w=args.img_w,
        flow_dir=args.flow_dir,
        video_folders=args.video_folders,
        max_videos=args.max_videos,
        n_max=args.n_max_objects,
        c_z=args.c_z,
        h_z=args.h_z,
        w_z=args.w_z,
        random_start=False,   # always start from frame 0 for evaluation
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    all_psnr, all_ssim, all_lpips = [], [], []

    for batch in tqdm(loader, desc="Inference"):
        frames_all = batch["frames"].to(device)   # (1, T, 3, H, W)
        video_idx = batch["video_idx"][0]
        B, T, C, H, W = frames_all.shape

        if T < T_needed:
            continue

        frames_flat = frames_all[0]  # (T, 3, H, W)

        # Encode all frames to latents
        with torch.no_grad():
            z_flat, _, _ = vae.encode(frames_flat)  # (T, C_z, H_z, W_z)

        num_input = args.num_input_frames
        num_pred  = args.num_pred_frames

        # ── Optional TTT ──────────────────────────────────────────────────────
        if args.ttt:
            temporal_used = run_ttt(
                temporal, vae,
                z_inputs=z_flat[:num_input].unsqueeze(0),
                frames_inputs=frames_flat[:num_input].unsqueeze(0),
                ttt_steps=args.ttt_steps,
                ttt_lr=args.ttt_lr,
                device=device,
            )
        else:
            temporal_used = temporal

        # ── Predict ───────────────────────────────────────────────────────────
        temporal_used.eval()
        with torch.no_grad():
            pred_frames = predict_video(
                vae, temporal_used,
                frames=frames_flat,
                z_all=z_flat,
                num_input=num_input,
                num_pred=num_pred,
                device=device,
            )  # (num_pred, 3, H, W)

        # Ground-truth frames for comparison
        gt_frames = frames_flat[num_input: num_input + num_pred]  # (num_pred, 3, H, W)

        # ── Metrics ───────────────────────────────────────────────────────────
        psnr = compute_psnr(pred_frames, gt_frames)
        ssim = compute_ssim(pred_frames.unsqueeze(0), gt_frames.unsqueeze(0))
        lpips_val = compute_lpips(pred_frames, gt_frames)

        all_psnr.append(psnr)
        all_ssim.append(ssim)
        all_lpips.append(lpips_val)

        # ── Save predicted MP4 ────────────────────────────────────────────────
        out_path = os.path.join(args.output_dir, f"video_{video_idx:05d}_pred.mp4")
        save_mp4(_to_uint8_nhwc(pred_frames), out_path)

    # ── Aggregate metrics ──────────────────────────────────────────────────────
    mean_psnr  = float(np.mean(all_psnr))  if all_psnr  else 0.0
    mean_ssim  = float(np.mean(all_ssim))  if all_ssim  else 0.0
    mean_lpips = float(np.mean(all_lpips)) if all_lpips else 0.0

    print(f"\n── Evaluation Results ──────────────────────────────────")
    print(f"  Videos evaluated : {len(all_psnr)}")
    print(f"  PSNR  (↑)        : {mean_psnr:.2f} dB")
    print(f"  SSIM  (↑)        : {mean_ssim:.4f}")
    print(f"  LPIPS (↓)        : {mean_lpips:.4f}")

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "n_videos": len(all_psnr),
            "psnr":  mean_psnr,
            "ssim":  mean_ssim,
            "lpips": mean_lpips,
            "per_video": {
                "psnr":  all_psnr,
                "ssim":  all_ssim,
                "lpips": all_lpips,
            },
        }, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
