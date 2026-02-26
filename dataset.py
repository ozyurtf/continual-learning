"""
dataset.py — VideoDataset for CLEVRER video prediction.

Each __getitem__ returns a dict with a subsequence of length seq_len from one video:
    frames          : (T, 3, H, W)          float32 [0,1]
    latents         : (T, C_z, H_z, W_z)   float32  (or None)
    flows           : (T-1, 2, H_z, W_z)   float32  (or None)
    occ_masks       : (T-1, 1, H_z, W_z)   float32  (or None)
    physics_flows   : (T-1, 2, H_z, W_z)   float32  (or None)
    obj_states      : (T, N_max, 6)         float32  (or None)
    collision_labels: (T-1,)                float32
    video_idx       : int
"""

import os
import glob
import json
import random
import struct
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.io as tv_io
import torchvision.transforms.functional as TF


# Per-split folder names used by CLEVRER
_SPLIT_DIRS = {
    "train": "video_train",
    "val":   "video_validation",
    "test":  "video_test",
}

_ANNOT_DIRS = {
    "train": "annotation_train",
    "val":   "annotation_validation",
}


def _find_mp4s(data_root: str, split: str,
               video_folders=None, max_videos: int = None) -> list[str]:
    """Return sorted list of MP4 paths for the given split, optionally capped."""
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


def _video_idx_from_path(mp4_path: str) -> int:
    """Extract integer index from filename like video_00042.mp4."""
    stem = os.path.splitext(os.path.basename(mp4_path))[0]  # "video_00042"
    return int(stem.split("_")[1])


def _find_annotation(data_root: str, split: str, video_idx: int) -> str | None:
    """Return annotation JSON path or None if not available (e.g. test split)."""
    if split not in _ANNOT_DIRS:
        return None
    annot_base = os.path.join(data_root, _ANNOT_DIRS[split])
    # Annotations are in folders like annotation_00000-01000
    for folder in sorted(os.listdir(annot_base)):
        path = os.path.join(annot_base, folder, f"annotation_{video_idx:05d}.json")
        if os.path.exists(path):
            return path
    return None


def _load_annotation(path: str, n_frames: int,
                      n_max: int = 6, state_dim: int = 6):
    """
    Load annotation JSON and return:
        obj_states      : np.ndarray (n_frames, n_max, state_dim)
                          state_dim = 6: x, y, z, vx, vy, vz  (normalised later)
        collision_labels: np.ndarray (n_frames - 1,) binary float32
                          label[t] = 1 if a collision occurs between frame t and t+1
    """
    with open(path) as f:
        ann = json.load(f)

    # ── Object states ──────────────────────────────────────────────────────
    obj_states = np.zeros((n_frames, n_max, state_dim), dtype=np.float32)
    for frame_data in ann.get("motion_trajectory", []):
        t = frame_data["frame_id"]
        if t >= n_frames:
            continue
        for obj in frame_data.get("objects", []):
            oid = obj["object_id"]
            if oid >= n_max:
                continue
            loc = obj.get("location", [0.0, 0.0, 0.0])
            vel = obj.get("velocity", [0.0, 0.0, 0.0])
            obj_states[t, oid, :3] = loc[:3]
            obj_states[t, oid, 3:6] = vel[:3]

    # ── Collision labels ───────────────────────────────────────────────────
    collision_labels = np.zeros(n_frames - 1, dtype=np.float32)
    for col in ann.get("collision", []):
        frame_id = col["frame_id"]
        # label at index t means "collision happens going from frame t to t+1"
        if 0 <= frame_id < n_frames - 1:
            collision_labels[frame_id] = 1.0

    return obj_states, collision_labels


def _load_frames(mp4_path: str, img_h: int, img_w: int) -> torch.Tensor:
    """
    Decode all frames from an MP4 and return (T, 3, H, W) float32 in [0, 1].
    Falls back to cv2 if torchvision cannot read the file.
    """
    try:
        frames, _, _ = tv_io.read_video(mp4_path, pts_unit="sec", output_format="TCHW")
        # frames: (T, C, H, W) uint8
        frames = frames.float() / 255.0
    except Exception:
        import cv2
        cap = cv2.VideoCapture(mp4_path)
        buf = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # cv2 reads BGR; convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buf.append(frame_rgb)
        cap.release()
        arr = np.stack(buf, axis=0).astype(np.float32) / 255.0  # (T, H, W, 3)
        frames = torch.from_numpy(arr).permute(0, 3, 1, 2)       # (T, 3, H, W)

    # Resize if needed
    T, C, H, W = frames.shape
    if H != img_h or W != img_w:
        frames = TF.resize(frames, [img_h, img_w], antialias=True)

    return frames


class VideoDataset(Dataset):
    """
    Loads CLEVRER video subsequences for training or inference.

    Parameters
    ----------
    data_root     : project root directory
    split         : 'train' | 'val' | 'test'
    seq_len       : number of frames per sample (T)
    img_h, img_w  : resize target
    flow_dir      : directory with precomputed .npy arrays
    video_folders : list of subfolder names to restrict to (None = all)
    n_max         : max objects per scene
    c_z           : VAE latent channels (used if latents are precomputed)
    h_z, w_z      : latent spatial dimensions (used if latents are precomputed)
    random_start  : if True, pick a random start frame; else start from 0
    """

    def __init__(self, data_root: str, split: str, seq_len: int = 10,
                 img_h: int = 320, img_w: int = 480,
                 flow_dir: str = "precomputed",
                 video_folders=None, max_videos: int = None,
                 n_max: int = 6, c_z: int = 32, h_z: int = 40, w_z: int = 60,
                 random_start: bool = True):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.seq_len = seq_len
        self.img_h = img_h
        self.img_w = img_w
        self.flow_dir = flow_dir
        self.n_max = n_max
        self.c_z = c_z
        self.h_z = h_z
        self.w_z = w_z
        self.random_start = random_start

        self.mp4_paths = _find_mp4s(data_root, split, video_folders, max_videos)
        if not self.mp4_paths:
            raise FileNotFoundError(
                f"No MP4 files found for split '{split}' "
                f"(video_folders={video_folders}) under {data_root}"
            )

    def __len__(self) -> int:
        return len(self.mp4_paths)

    def __getitem__(self, idx: int) -> dict:
        mp4_path = self.mp4_paths[idx]
        video_idx = _video_idx_from_path(mp4_path)

        # ── Frames ────────────────────────────────────────────────────────────
        frames = _load_frames(mp4_path, self.img_h, self.img_w)
        T_full = frames.shape[0]

        # ── Subsequence window ────────────────────────────────────────────────
        T = min(self.seq_len, T_full)
        if self.random_start and T_full > T:
            start = random.randint(0, T_full - T)
        else:
            start = 0
        end = start + T
        frames = frames[start:end]  # (T, 3, H, W)

        # ── Precomputed arrays ─────────────────────────────────────────────────
        prefix = os.path.join(self.flow_dir, f"video_{video_idx:05d}")

        def load_npy(suffix, expected_shape_0):
            path = f"{prefix}_{suffix}.npy"
            if not os.path.exists(path):
                return None
            arr = np.load(path)[start:start + expected_shape_0]
            return torch.from_numpy(arr)

        flows         = load_npy("flow",         T - 1)  # (T-1, 2, H_z, W_z)
        occ_masks     = load_npy("occ",          T - 1)  # (T-1, 1, H_z, W_z)
        physics_flows = load_npy("physics_flow", T - 1)  # (T-1, 2, H_z, W_z)
        latents       = load_npy("latents",      T)      # (T, C_z, H_z, W_z)

        # ── Annotations ───────────────────────────────────────────────────────
        annot_path = _find_annotation(self.data_root, self.split, video_idx)
        if annot_path is not None:
            obj_states_full, col_labels_full = _load_annotation(
                annot_path, T_full, self.n_max
            )
            obj_states      = torch.from_numpy(obj_states_full[start:end])
            collision_labels = torch.from_numpy(col_labels_full[start:end - 1])
        else:
            obj_states       = None
            collision_labels = torch.zeros(T - 1, dtype=torch.float32)

        return {
            "frames":           frames,           # (T, 3, H, W)
            "latents":          latents,           # (T, C_z, H_z, W_z) or None
            "flows":            flows,             # (T-1, 2, H_z, W_z) or None
            "occ_masks":        occ_masks,         # (T-1, 1, H_z, W_z) or None
            "physics_flows":    physics_flows,     # (T-1, 2, H_z, W_z) or None
            "obj_states":       obj_states,        # (T, N_max, 6) or None
            "collision_labels": collision_labels,  # (T-1,)
            "video_idx":        video_idx,
        }


def collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate that handles None tensors gracefully.
    Stacks tensors across the batch; None fields become None if *all* samples
    are None, otherwise missing samples are skipped for that field.
    """
    keys = batch[0].keys()
    result = {}
    for key in keys:
        if key == "video_idx":
            result[key] = [s[key] for s in batch]
            continue
        values = [s[key] for s in batch if s[key] is not None]
        if not values:
            result[key] = None
        elif len(values) == len(batch):
            result[key] = torch.stack(values, dim=0)
        else:
            # Partial batch: return None to signal unavailability
            result[key] = None
    return result
