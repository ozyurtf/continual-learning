import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.io as tv_io
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.utils import flow_to_image

from models import VAE
from utils import load_checkpoint


def backward_warp(feat: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    feat : (B, C, H, W)
    flow : (B, 2, H, W)  forward flow in pixel units
    Returns: (B, C, H, W)
    """
    B, C, H, W = feat.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=feat.dtype, device=feat.device),
        torch.arange(W, dtype=feat.dtype, device=feat.device),
        indexing="ij",
    )
    grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
    sample_grid = grid + flow
    sample_grid[:, 0] = 2.0 * sample_grid[:, 0] / max(W - 1, 1) - 1.0
    sample_grid[:, 1] = 2.0 * sample_grid[:, 1] / max(H - 1, 1) - 1.0
    sample_grid = sample_grid.permute(0, 2, 3, 1)
    return F.grid_sample(feat, sample_grid, mode="bilinear",
                         padding_mode="border", align_corners=True)


def warp_with_flow(frame_uint8: torch.Tensor, flow_bwd: torch.Tensor) -> np.ndarray:
    """
    frame_uint8 : (3, H, W) uint8
    flow_bwd    : (2, H, W) backward flow in pixel units (t+1 → t)
    Returns     : (H, W, 3) float32 [0, 1]
    """
    frame_f = frame_uint8.float().unsqueeze(0) / 255.0
    warped  = backward_warp(frame_f, flow_bwd.unsqueeze(0))
    return warped[0].permute(1, 2, 0).clamp(0, 1).numpy()


def load_frames(mp4_path: str, img_h: int, img_w: int) -> torch.Tensor:
    """Returns (T, 3, H, W) uint8."""
    try:
        frames, _, _ = tv_io.read_video(mp4_path, pts_unit="sec", output_format="TCHW")
    except Exception:
        import cv2
        cap = cv2.VideoCapture(mp4_path)
        buf = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (img_w, img_h), interpolation=cv2.INTER_AREA)
            buf.append(frame_rgb)
        cap.release()
        return torch.from_numpy(np.stack(buf).astype(np.uint8)).permute(0, 3, 1, 2)

    T, C, H, W = frames.shape
    if H != img_h or W != img_w:
        frames = TF.resize(frames, [img_h, img_w], antialias=True)
    return frames


def find_video(data_root: str, split: str = "test", video_idx: int = None) -> str:
    split_dirs = {"train": "video_train", "val": "video_validation", "test": "video_test"}
    base = os.path.join(data_root, split_dirs[split])
    if video_idx is not None:
        pattern = os.path.join(base, "**", f"video_{video_idx:05d}.mp4")
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            raise FileNotFoundError(
                f"video_{video_idx:05d}.mp4 not found under {base}"
            )
        return matches[0]
    # No index given — return the first video found alphabetically
    matches = sorted(glob.glob(os.path.join(base, "**", "*.mp4"), recursive=True))
    if not matches:
        raise FileNotFoundError(f"No .mp4 files found under {base}")
    return matches[0]


def build_raft_preprocess(img_h: int, img_w: int):
    return T.Compose([
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=0.5, std=0.5),
        T.Resize(size=(img_h, img_w), antialias=False),
    ])


@torch.no_grad()
def compute_flows(raft, preprocess, frames_uint8: torch.Tensor,
                  device: torch.device, timesteps: list):
    """Compute forward and backward flow for each t in timesteps.
    Returns:
        flows_fwd : (N, 2, H, W) — RAFT(t → t+1)
        flows_bwd : (N, 2, H, W) — RAFT(t+1 → t)
    """
    fwd, bwd = [], []
    for t in timesteps:
        f1 = preprocess(frames_uint8[t:t+1]).to(device)
        f2 = preprocess(frames_uint8[t+1:t+2]).to(device)
        fwd.append(raft(f1, f2)[-1][0].cpu())
        bwd.append(raft(f2, f1)[-1][0].cpu())
    return torch.stack(fwd, dim=0), torch.stack(bwd, dim=0)  # each (N, 2, H, W)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae_checkpoint", required=True)
    p.add_argument("--data_root", default=".")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--video_idx", type=int, default=None,
                   help="Video index to load (e.g. 15000). If not set, uses the first video found.")
    p.add_argument("--img_h", type=int, default=160)
    p.add_argument("--img_w", type=int, default=240)
    p.add_argument("--c_z", type=int, default=32)
    p.add_argument("--stride", type=int, default=1,
                   help="Take every Nth frame (e.g. 5 → frames 0,5,10,...)")
    p.add_argument("--timesteps", default="0,10,20,30")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="vae_warp.png")
    args = p.parse_args()

    device = torch.device(args.device)
    timesteps = [int(t) for t in args.timesteps.split(",")]

    mp4_path = find_video(args.data_root, args.split, args.video_idx)
    video_name = os.path.splitext(os.path.basename(mp4_path))[0]  # e.g. "video_15000"
    print(f"Video : {mp4_path}")
    frames_uint8 = load_frames(mp4_path, args.img_h, args.img_w)  # (T, 3, H, W) uint8
    if args.stride > 1:
        frames_uint8 = frames_uint8[::args.stride]
        print(f"Subsampled every {args.stride} frames → {frames_uint8.shape[0]} frames")
    frames_float = frames_uint8.float() / 255.0                    # (T, 3, H, W) [0,1]
    print(f"Frames: {frames_uint8.shape}")

    max_t = frames_uint8.shape[0] - 2
    timesteps = [t for t in timesteps if t <= max_t]
    n = len(timesteps)

    print(f"Loading VAE from {args.vae_checkpoint} …")
    vae = VAE(c_z=args.c_z).to(device)
    load_checkpoint(args.vae_checkpoint, vae, device=device)
    vae.eval()

    # Reconstruct all frames needed
    all_t = sorted(set(timesteps + [t + 1 for t in timesteps]))
    with torch.no_grad():
        batch = frames_float[all_t].to(device)       # (N, 3, H, W)
        z, _, _ = vae.encode(batch)                  # (N, C_z, H_z, W_z)
        recon = vae.decode(z).clamp(0, 1).cpu()      # (N, 3, H, W)
    recon_map = {t: recon[i] for i, t in enumerate(all_t)}

    print("Loading RAFT Large …")
    raft = raft_large(weights=Raft_Large_Weights.DEFAULT).eval().to(device)
    preprocess = build_raft_preprocess(args.img_h, args.img_w)

    print("Computing flows (gap=1, on original frames) …")
    flows_fwd, flows_bwd = compute_flows(raft, preprocess, frames_uint8, device, timesteps)

    global_max = float(flows_fwd.norm(dim=1).max())
    print(f"Global max flow magnitude: {global_max:.2f}px")

    # flow_to_image: (N, 2, H, W) → (N, 3, H, W) uint8
    flow_imgs = flow_to_image(flows_fwd / global_max)

    # Rows (per column = one timestep):
    #   0 : Original frame t
    #   1 : VAE reconstructed frame t
    #   2 : RAFT Large flow  (t → t+1)
    #   3 : Warped reconstructed frame  (pred. t+1)
    #   4 : Ground truth frame t+1
    row_labels = [
        "Original (t)",
        "VAE reconstructed (t)",
        "RAFT Large flow (t→t+1)",
        "Warped recon (pred. t+1)",
        "Ground truth (t+1)",
    ]
    nrows = len(row_labels)

    fig, axes = plt.subplots(nrows, n, figsize=(4 * n, 3.2 * nrows))
    fig.suptitle(
        f"VAE reconstruction + warping — {video_name}  "
        f"({args.img_h}×{args.img_w})  split={args.split}",
        fontsize=12,
    )

    if n == 1:
        axes = axes[:, None]

    def label(ax, text):
        ax.text(0.01, 0.97, text, transform=ax.transAxes,
                fontsize=8, va="top", ha="left", color="white", fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.6, pad=3, boxstyle="round"))

    for col, t in enumerate(timesteps):
        orig_np  = frames_uint8[t].permute(1, 2, 0).numpy()
        recon_np = recon_map[t].permute(1, 2, 0).numpy()
        gt_np    = frames_uint8[t + 1].permute(1, 2, 0).numpy()

        flow_np   = np.asarray(TF.to_pil_image(flow_imgs[col].cpu()))

        # Warp the VAE reconstructed frame using backward flow (t+1 → t)
        recon_uint8 = (recon_map[t] * 255).clamp(0, 255).byte()
        warped_np   = warp_with_flow(recon_uint8, flows_bwd[col])

        err = np.abs(warped_np - gt_np.astype(np.float32) / 255.0).mean()

        axes[0, col].imshow(orig_np)
        axes[1, col].imshow(recon_np)
        axes[2, col].imshow(flow_np)
        axes[3, col].imshow(warped_np)
        axes[4, col].imshow(gt_np)

        axes[0, col].set_title(f"t={t}", fontsize=9)
        axes[1, col].set_title(f"t={t}", fontsize=9)
        axes[2, col].set_title(f"max={flows_bwd[col].norm(dim=0).max():.1f}px", fontsize=9)
        axes[3, col].set_title(f"L1={err:.4f}", fontsize=9)
        axes[4, col].set_title(f"t+1={t+1}", fontsize=9)

    for ax in axes.flat:
        ax.axis("off")

    for row, text in enumerate(row_labels):
        label(axes[row, 0], text)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
