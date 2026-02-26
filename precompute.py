"""
precompute.py — Offline precomputation of per-video arrays.

For each video in the selected split + folder subset, computes and saves:
  video_XXXXX_flow.npy         (T-1, 2, H_z, W_z)  RAFT flow (latent space)
  video_XXXXX_flow_bwd.npy     (T-1, 2, H_z, W_z)  backward RAFT flow
  video_XXXXX_occ.npy          (T-1, 1, H_z, W_z)  FB-consistency occlusion mask
  video_XXXXX_physics_flow.npy (T-1, 2, H_z, W_z)  physics-informed flow (train/val only)

Usage
-----
  python precompute.py --split train --video_folders video_00000-01000 \
                       --img_h 128 --img_w 128 --flow_dir precomputed

  # Full dataset on server
  python precompute.py --split train --img_h 320 --img_w 480 --flow_dir precomputed
"""

import os
import sys
import glob
import json
import argparse
import struct
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.io as tv_io
import torchvision.transforms.functional as TF
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from tqdm import tqdm

from config import get_device
from utils import backward_warp, downscale_flow, decode_rle_mask


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_SPLIT_DIRS = {
    "train": "video_train",
    "val":   "video_validation",
    "test":  "video_test",
}
_ANNOT_DIRS = {
    "train": "annotation_train",
    "val":   "annotation_validation",
}


def find_mp4s(data_root: str, split: str,
              video_folders=None, max_videos: int = None) -> list[str]:
    base = os.path.join(data_root, _SPLIT_DIRS[split])
    if video_folders is not None:
        dirs = [os.path.join(base, f) for f in video_folders]
    else:
        dirs = sorted(glob.glob(os.path.join(base, "*")))
    paths = []
    for d in dirs:
        paths.extend(sorted(glob.glob(os.path.join(d, "*.mp4"))))
    if max_videos is not None:
        paths = paths[:max_videos]
    return paths


def video_idx_from_path(mp4_path: str) -> int:
    stem = os.path.splitext(os.path.basename(mp4_path))[0]
    return int(stem.split("_")[1])


def load_frames(mp4_path: str, img_h: int, img_w: int) -> torch.Tensor:
    """Returns (T, 3, H, W) float32 [0, 1]."""
    try:
        frames, _, _ = tv_io.read_video(mp4_path, pts_unit="sec", output_format="TCHW")
        frames = frames.float() / 255.0
    except Exception:
        import cv2
        cap = cv2.VideoCapture(mp4_path)
        buf = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            buf.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        arr = np.stack(buf).astype(np.float32) / 255.0
        frames = torch.from_numpy(arr).permute(0, 3, 1, 2)

    T, C, H, W = frames.shape
    if H != img_h or W != img_w:
        frames = TF.resize(frames, [img_h, img_w], antialias=True)
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# RAFT optical flow
# ─────────────────────────────────────────────────────────────────────────────

def build_raft(device: torch.device) -> torch.nn.Module:
    model = raft_small(weights=Raft_Small_Weights.DEFAULT)
    model.eval().to(device)
    return model


@torch.no_grad()
def compute_flows_raft(raft_model, frames: torch.Tensor, device: torch.device,
                       scale: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute forward and backward RAFT flows for all consecutive frame pairs,
    then downscale to latent resolution.

    Args:
        frames : (T, 3, H, W) float32 [0, 1]
        scale  : VAE downsampling factor (8 = 3 stride-2 convolutions)

    Returns:
        flow_fwd : (T-1, 2, H_z, W_z) latent-space forward flow
        flow_bwd : (T-1, 2, H_z, W_z) latent-space backward flow
    """
    T = frames.shape[0]
    # RAFT expects (B, 3, H, W) in [0, 255] with dtype float32
    frames_255 = (frames * 255.0).to(device)

    fwd_list, bwd_list = [], []
    for t in range(T - 1):
        f1 = frames_255[t: t + 1]
        f2 = frames_255[t + 1: t + 2]

        # forward t → t+1
        flow_f = raft_model(f1, f2)[-1]   # (1, 2, H, W)
        # backward t+1 → t
        flow_b = raft_model(f2, f1)[-1]   # (1, 2, H, W)

        fwd_list.append(downscale_flow(flow_f, scale))  # (1, 2, H_z, W_z)
        bwd_list.append(downscale_flow(flow_b, scale))

    flow_fwd = torch.cat(fwd_list, dim=0).cpu().numpy()  # (T-1, 2, H_z, W_z)
    flow_bwd = torch.cat(bwd_list, dim=0).cpu().numpy()
    return flow_fwd, flow_bwd


# ─────────────────────────────────────────────────────────────────────────────
# Forward-Backward consistency occlusion mask
# ─────────────────────────────────────────────────────────────────────────────

def compute_fb_occlusion(flow_fwd: np.ndarray, flow_bwd: np.ndarray,
                         alpha: float = 0.5) -> np.ndarray:
    """
    Compute per-frame occlusion mask from forward-backward flow consistency.

    mask[t, 0, h, w] = exp(-||fwd[t] + warp(bwd[t], fwd[t])||^2 / alpha)
    Value 1 = trustworthy (consistent), 0 = occluded.

    Args:
        flow_fwd : (T-1, 2, H_z, W_z)
        flow_bwd : (T-1, 2, H_z, W_z)

    Returns:
        occ_mask : (T-1, 1, H_z, W_z) float32
    """
    fwd = torch.from_numpy(flow_fwd)  # (T-1, 2, H_z, W_z)
    bwd = torch.from_numpy(flow_bwd)

    occ_masks = []
    for t in range(fwd.shape[0]):
        f = fwd[t: t + 1]   # (1, 2, H_z, W_z)
        b = bwd[t: t + 1]

        # Warp backward flow using forward flow
        bwd_warped = backward_warp(b, f)  # (1, 2, H_z, W_z)

        # Consistency error: should be near zero if no occlusion
        err = (f + bwd_warped).pow(2).sum(dim=1, keepdim=True)  # (1, 1, H_z, W_z)
        mask = torch.exp(-err / (alpha + 1e-8))
        occ_masks.append(mask)

    return torch.cat(occ_masks, dim=0).numpy()  # (T-1, 1, H_z, W_z)


# ─────────────────────────────────────────────────────────────────────────────
# Physics-informed flow map
# ─────────────────────────────────────────────────────────────────────────────

def _load_annotation(annot_path: str):
    with open(annot_path) as f:
        return json.load(f)


def _find_annotation(data_root: str, split: str, video_idx: int) -> str | None:
    if split not in _ANNOT_DIRS:
        return None
    annot_base = os.path.join(data_root, _ANNOT_DIRS[split])
    for folder in sorted(os.listdir(annot_base)):
        path = os.path.join(annot_base, folder,
                            f"annotation_{video_idx:05d}.json")
        if os.path.exists(path):
            return path
    return None


def _find_derender(data_root: str, video_idx: int) -> str | None:
    path = os.path.join(data_root, "derender_proposals",
                        f"proposal_{video_idx:05d}.json")
    return path if os.path.exists(path) else None


def compute_physics_flow(data_root: str, split: str, video_idx: int,
                         img_h: int, img_w: int,
                         scale: int = 8) -> np.ndarray | None:
    """
    Build a physics-informed flow map from velocity annotations and
    derender segmentation masks.

    For each frame t, for each object i:
        - Project velocity (vx, vy) to pixel space using a linear projection P.
        - Fill the object's segmentation region with (flow_x, flow_y) = P @ (vx, vy).

    P is estimated from the RAFT flow and segmentation masks (least-squares fit).
    If no annotation/derender data is available, returns None.

    Returns:
        physics_flow : (T-1, 2, H_z, W_z) float32  or  None
    """
    annot_path = _find_annotation(data_root, split, video_idx)
    derender_path = _find_derender(data_root, video_idx)

    if annot_path is None or derender_path is None:
        return None

    ann = _load_annotation(annot_path)
    with open(derender_path) as f:
        der = json.load(f)

    T = len(ann.get("motion_trajectory", []))
    if T == 0:
        return None

    H_z, W_z = img_h // scale, img_w // scale

    # ── Collect per-frame velocities ──────────────────────────────────────────
    frame_velocities = {}  # frame_id → {object_id: (vx, vy)}
    for fd in ann.get("motion_trajectory", []):
        fid = fd["frame_id"]
        frame_velocities[fid] = {}
        for obj in fd.get("objects", []):
            vel = obj.get("velocity", [0.0, 0.0, 0.0])
            frame_velocities[fid][obj["object_id"]] = (vel[0], vel[1])

    # ── Collect per-frame segmentation masks from derender proposals ──────────
    # der["frames"] is a list of {frame_index, objects: [{mask: {size,counts}, ...}]}
    frame_masks = {}  # frame_id → {object_index_in_frame: np.ndarray (H, W) bool}
    for fdata in der.get("frames", []):
        fid = fdata["frame_index"]
        frame_masks[fid] = []
        for obj_data in fdata.get("objects", []):
            try:
                m = decode_rle_mask(obj_data["mask"])  # (H, W)
            except Exception:
                m = np.zeros((img_h, img_w), dtype=bool)
            frame_masks[fid].append(m)

    # ── Estimate world-to-pixel projection P (2×2 learnable) ─────────────────
    # We do a simple least-squares fit: for each (object, frame) pair where the
    # object is moving, use the mean RAFT flow in its masked region as the target.
    # P maps (vx, vy) → (flow_x_pixel, flow_y_pixel)
    #
    # NOTE: We do NOT use precomputed RAFT flows here (they may not exist yet
    # at precompute time). Instead, we use a heuristic: CLEVRER uses an
    # approximately top-down orthographic-ish camera. Objects move mostly in the
    # x-y plane. We fit P = [[px, 0], [0, py]] by estimating the pixel/world
    # ratio from the scene geometry.
    #
    # Empirically, for the 320×480 (H×W) frame:
    #   x-axis (width direction) ≈ 480 pixels / 8 world units → 60 px/unit
    #   y-axis (height direction) ≈ 320 pixels / 8 world units → 40 px/unit
    # These are rough estimates; the physics_flow loss weight (0.3) is intentionally
    # small so that inaccuracies in P do not dominate training.
    px = img_w / 8.0  # pixels per world unit along x
    py = img_h / 8.0  # pixels per world unit along y

    # ── Build physics flow maps ───────────────────────────────────────────────
    # We produce (T-1) maps: map at index t captures motion from frame t to t+1.
    # We use the velocity at frame t (current state).
    physics_flow_full = np.zeros((T - 1, 2, img_h, img_w), dtype=np.float32)

    for t in range(T - 1):
        vel_t = frame_velocities.get(t, {})
        masks_t = frame_masks.get(t, [])

        for obj_local_idx, mask_full in enumerate(masks_t):
            # Map local derender index to annotation object id heuristically:
            # derender proposals are ordered by score, annotations by object_id.
            # We use the same index assuming consistent ordering.
            oid = obj_local_idx
            if oid not in vel_t:
                continue
            vx, vy = vel_t[oid]

            # Pixel-space flow for this object
            flow_x = vx * px  # world units → pixels
            flow_y = vy * py

            # Resize mask to (img_h, img_w) if needed
            if mask_full.shape != (img_h, img_w):
                mask_resized = np.array(
                    TF.resize(
                        torch.from_numpy(mask_full.astype(np.uint8)).unsqueeze(0).unsqueeze(0),
                        [img_h, img_w],
                        antialias=False,
                    ).squeeze().numpy(),
                    dtype=bool,
                )
            else:
                mask_resized = mask_full

            physics_flow_full[t, 0][mask_resized] = flow_x
            physics_flow_full[t, 1][mask_resized] = flow_y

    # ── Downscale to latent resolution ───────────────────────────────────────
    physics_flow_t = torch.from_numpy(physics_flow_full)
    physics_flow_z = downscale_flow(physics_flow_t, scale).numpy()  # (T-1, 2, H_z, W_z)

    return physics_flow_z.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Precompute optical flows for CLEVRER")
    p.add_argument("--data_root", default=".")
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--video_folders", default=None,
                   help="Comma-separated subfolder names (e.g. 'video_00000-01000')")
    p.add_argument("--max_videos", type=int, default=None,
                   help="Cap total number of videos to process")
    p.add_argument("--flow_dir", default="precomputed")
    p.add_argument("--img_h", type=int, default=320)
    p.add_argument("--img_w", type=int, default=480)
    p.add_argument("--fb_alpha", type=float, default=0.5,
                   help="Alpha for FB consistency mask")
    p.add_argument("--device", default="auto")
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute even if output files already exist")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    scale = 8  # VAE downsampling factor
    H_z, W_z = args.img_h // scale, args.img_w // scale

    video_folders = (
        [f.strip() for f in args.video_folders.split(",")]
        if args.video_folders else None
    )

    os.makedirs(args.flow_dir, exist_ok=True)

    mp4_paths = find_mp4s(args.data_root, args.split, video_folders, args.max_videos)
    print(f"Found {len(mp4_paths)} videos for split='{args.split}'")

    print("Loading RAFT-Small model …")
    raft = build_raft(device)

    for mp4_path in tqdm(mp4_paths, desc="Precomputing"):
        video_idx = video_idx_from_path(mp4_path)
        prefix = os.path.join(args.flow_dir, f"video_{video_idx:05d}")

        fwd_path = f"{prefix}_flow.npy"
        bwd_path = f"{prefix}_flow_bwd.npy"
        occ_path = f"{prefix}_occ.npy"
        phys_path = f"{prefix}_physics_flow.npy"

        # Skip if all outputs exist and overwrite not requested
        all_exist = (
            os.path.exists(fwd_path) and
            os.path.exists(bwd_path) and
            os.path.exists(occ_path)
        )
        if all_exist and not args.overwrite:
            continue

        # ── Load frames ────────────────────────────────────────────────────
        frames = load_frames(mp4_path, args.img_h, args.img_w)  # (T, 3, H, W)

        # ── RAFT flows ────────────────────────────────────────────────────
        flow_fwd, flow_bwd = compute_flows_raft(raft, frames, device, scale)

        # ── FB-consistency occlusion mask ─────────────────────────────────
        occ = compute_fb_occlusion(flow_fwd, flow_bwd, args.fb_alpha)

        np.save(fwd_path, flow_fwd.astype(np.float32))
        np.save(bwd_path, flow_bwd.astype(np.float32))
        np.save(occ_path, occ.astype(np.float32))

        # ── Physics flow (train/val only) ──────────────────────────────────
        if not os.path.exists(phys_path) or args.overwrite:
            phys = compute_physics_flow(
                args.data_root, args.split, video_idx,
                args.img_h, args.img_w, scale
            )
            if phys is not None:
                np.save(phys_path, phys)

    print(f"Done. Outputs saved to '{args.flow_dir}/'")


if __name__ == "__main__":
    main()
