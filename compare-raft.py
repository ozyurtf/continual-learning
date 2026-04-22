import os
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.io as tv_io
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchvision.models.optical_flow import (
    raft_small, Raft_Small_Weights,
    raft_large, Raft_Large_Weights,
)
from torchvision.utils import flow_to_image


def backward_warp(feat: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Warp feat using flow via bilinear sampling.
    feat : (B, C, H, W)
    flow : (B, 2, H, W)  — flow[0]=dx (W direction), flow[1]=dy (H direction)
    Returns: (B, C, H, W)
    """
    B, C, H, W = feat.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=feat.dtype, device=feat.device),
        torch.arange(W, dtype=feat.dtype, device=feat.device),
        indexing="ij",
    )
    grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # (1, 2, H, W)
    sample_grid = grid + flow
    sample_grid[:, 0] = 2.0 * sample_grid[:, 0] / max(W - 1, 1) - 1.0
    sample_grid[:, 1] = 2.0 * sample_grid[:, 1] / max(H - 1, 1) - 1.0
    sample_grid = sample_grid.permute(0, 2, 3, 1)
    return F.grid_sample(feat, sample_grid, mode="bilinear",
                         padding_mode="border", align_corners=True)


def load_frames(mp4_path: str, img_h: int, img_w: int) -> torch.Tensor:
    """Returns (T, 3, H, W) uint8 in [0, 255]."""
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
        arr = np.stack(buf).astype(np.uint8)
        return torch.from_numpy(arr).permute(0, 3, 1, 2)

    T, C, H, W = frames.shape
    if H != img_h or W != img_w:
        frames = TF.resize(frames, [img_h, img_w], antialias=True)
    return frames  # uint8


def build_preprocess(img_h: int, img_w: int):
    """
    Preprocessing pipeline matching the torchvision RAFT tutorial:
      1. ConvertImageDtype(float32)  — uint8 [0,255] → float32 [0,1]
      2. Normalize(mean=0.5, std=0.5) — [0,1] → [-1,1]
      3. Resize to the display resolution
    """
    return T.Compose([
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=0.5, std=0.5),
        T.Resize(size=(img_h, img_w), antialias=False),
    ])


@torch.no_grad()
def run_raft(model, preprocess, frames_uint8: torch.Tensor,
             device: torch.device, timesteps: list, gap: int):
    """
    Compute forward (t→t+gap) and backward (t+gap→t) flows for each timestep t.
    Returns:
        flows_fwd : (N, 2, H, W)
        flows_bwd : (N, 2, H, W)
    """
    fwd, bwd = [], []
    for t in timesteps:
        f1 = preprocess(frames_uint8[t:t+1]).to(device)
        f2 = preprocess(frames_uint8[t+gap:t+gap+1]).to(device)
        fwd.append(model(f1, f2)[-1][0].cpu())
        bwd.append(model(f2, f1)[-1][0].cpu())
    return torch.stack(fwd, dim=0), torch.stack(bwd, dim=0)


def warp_with_flow(frame_uint8: torch.Tensor, flow_bwd: torch.Tensor) -> np.ndarray:
    """
    Approximate frame t+gap by warping frame t with backward flow (t+gap → t).
    frame_uint8 : (3, H, W) uint8
    flow_bwd    : (2, H, W) backward flow in pixel units
    Returns     : (H, W, 3) float32 [0, 1]
    """
    frame_f = frame_uint8.float().unsqueeze(0) / 255.0
    warped  = backward_warp(frame_f, flow_bwd.unsqueeze(0))
    return warped[0].permute(1, 2, 0).clamp(0, 1).numpy()


def find_video(data_root: str, video_idx: int = 0) -> str:
    pattern = os.path.join(data_root, "**", f"video_{video_idx:05d}.mp4")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        raise FileNotFoundError(f"video_{video_idx:05d}.mp4 not found under {data_root}")
    return matches[0]



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=".")
    p.add_argument("--img_h", type=int, default=320)
    p.add_argument("--img_w", type=int, default=480)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--timesteps", default="0,10,20,30",
                   help="Comma-separated frame indices t")
    p.add_argument("--gap", type=int, default=1,
                   help="Frame gap for flow computation (default: 5)")
    p.add_argument("--output", default="raft_comparison.png")
    args = p.parse_args()

    device = torch.device(args.device)
    timesteps = [int(t) for t in args.timesteps.split(",")]

    mp4_path = find_video(args.data_root, video_idx=0)
    print(f"Video : {mp4_path}")
    frames = load_frames(mp4_path, args.img_h, args.img_w)
    print(f"Frames: {frames.shape}  dtype={frames.dtype}")

    gap = args.gap
    max_t = frames.shape[0] - gap - 1
    timesteps = [t for t in timesteps if t <= max_t]
    n = len(timesteps)

    preprocess_small = build_preprocess(args.img_h, args.img_w)
    preprocess_large = build_preprocess(args.img_h, args.img_w)

    print("Loading RAFT Small …")
    model_small = raft_small(weights=Raft_Small_Weights.DEFAULT).eval().to(device)

    print("Loading RAFT Large …")
    model_large = raft_large(weights=Raft_Large_Weights.DEFAULT).eval().to(device)

    print(f"Running RAFT Small  (gap={gap}) …")
    flows_small_fwd, flows_small_bwd = run_raft(model_small, preprocess_small, frames, device, timesteps, gap)

    print(f"Running RAFT Large  (gap={gap}) …")
    flows_large_fwd, flows_large_bwd = run_raft(model_large, preprocess_large, frames, device, timesteps, gap)

    for i, t in enumerate(timesteps):
        ms = flows_small_fwd[i].norm(dim=0)
        ml = flows_large_fwd[i].norm(dim=0)
        print(f"t={t:3d}  Small: max={ms.max():.2f}px mean={ms.mean():.3f}px | "
              f"Large: max={ml.max():.2f}px mean={ml.mean():.3f}px")

    # Shared scale: normalize both models' flows by the same global max magnitude
    global_max = max(flows_small_fwd.norm(dim=1).max(), flows_large_fwd.norm(dim=1).max())
    print(f"Global max magnitude: {global_max:.2f}px  (shared color scale)")

    # flow_to_image: (N, 2, H, W) → (N, 3, H, W) uint8  (visualize forward flow)
    flow_imgs_small = flow_to_image(flows_small_fwd / global_max)
    flow_imgs_large = flow_to_image(flows_large_fwd / global_max)

    # Columns = timesteps, rows:
    #   0 : source frame t
    #   1 : target frame t+gap
    #   2 : RAFT Small flow
    #   3 : RAFT Large flow
    #   4 : Warped by RAFT Small
    #   5 : Warped by RAFT Large
    row_labels = [
        f"Source (t)",
        f"Target (t+{gap})",
        f"RAFT Small flow",
        f"RAFT Large flow",
        f"Warped — RAFT Small",
        f"Warped — RAFT Large",
        f"Target (t+{gap})",
    ]
    nrows = len(row_labels)

    fig, axes = plt.subplots(nrows, n, figsize=(4 * n, 3.2 * nrows))
    fig.suptitle(
        f"RAFT Small vs Large — video_00000  ({args.img_h}×{args.img_w})  gap={gap}",
        fontsize=13,
    )

    if n == 1:
        axes = axes[:, None]

    def label(ax, text):
        ax.text(0.01, 0.97, text, transform=ax.transAxes,
                fontsize=8, va="top", ha="left", color="white", fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.6, pad=3, boxstyle="round"))

    for col, t in enumerate(timesteps):
        # frames are uint8; convert to float [0,1] for display
        src_np = frames[t].permute(1, 2, 0).numpy().astype(np.float32) / 255.0
        gt_np  = frames[t + gap].permute(1, 2, 0).numpy().astype(np.float32) / 255.0

        # flow_to_image returns uint8 (3, H, W); use to_pil_image like the tutorial
        flow_small_np = np.asarray(TF.to_pil_image(flow_imgs_small[col].cpu()))
        flow_large_np = np.asarray(TF.to_pil_image(flow_imgs_large[col].cpu()))

        warped_small = warp_with_flow(frames[t], flows_small_bwd[col])
        warped_large = warp_with_flow(frames[t], flows_large_bwd[col])

        err_small = np.abs(warped_small - gt_np).mean()
        err_large = np.abs(warped_large - gt_np).mean()

        mag_small = flows_small_fwd[col].norm(dim=0)
        mag_large = flows_large_fwd[col].norm(dim=0)

        axes[0, col].imshow(src_np)
        axes[1, col].imshow(gt_np)
        axes[2, col].imshow(flow_small_np)
        axes[3, col].imshow(flow_large_np)
        axes[4, col].imshow(warped_small)
        axes[5, col].imshow(warped_large)
        axes[6, col].imshow(gt_np)

        axes[0, col].set_title(f"t={t}", fontsize=9)
        axes[1, col].set_title(f"t+{gap}={t+gap}", fontsize=9)
        axes[2, col].set_title(f"max={mag_small.max():.1f}px", fontsize=9)
        axes[3, col].set_title(f"max={mag_large.max():.1f}px", fontsize=9)
        axes[4, col].set_title(f"L1={err_small:.4f}", fontsize=9)
        axes[5, col].set_title(f"L1={err_large:.4f}", fontsize=9)
        axes[6, col].set_title(f"t+{gap}={t+gap}", fontsize=9)

    for ax in axes.flat:
        ax.axis("off")

    for row, text in enumerate(row_labels):
        label(axes[row, 0], text)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
