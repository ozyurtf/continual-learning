import os
import copy
import struct
import numpy as np
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Warping
# ─────────────────────────────────────────────────────────────────────────────

def backward_warp(feat: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Differentiable backward warping using bilinear interpolation.

    Args:
        feat : (B, C, H, W)  — feature map or latent to warp
        flow : (B, 2, H, W)  — flow in pixel/latent units; flow[0] = dx (W), flow[1] = dy (H)

    Returns:
        warped: (B, C, H, W)
    """
    B, C, H, W = feat.shape
    # Build normalised sampling grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=feat.dtype, device=feat.device),
        torch.arange(W, dtype=feat.dtype, device=feat.device),
        indexing="ij",
    )
    grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # (1, 2, H, W)

    # Add flow to base grid
    sample_grid = grid + flow  # (B, 2, H, W)

    # Normalise to [-1, 1]
    sample_grid[:, 0] = 2.0 * sample_grid[:, 0] / max(W - 1, 1) - 1.0
    sample_grid[:, 1] = 2.0 * sample_grid[:, 1] / max(H - 1, 1) - 1.0

    # grid_sample expects (B, H, W, 2) in (x, y) order
    sample_grid = sample_grid.permute(0, 2, 3, 1)
    warped = F.grid_sample(feat, sample_grid, mode="bilinear",
                           padding_mode="border", align_corners=True)
    return warped


def downscale_flow(flow: torch.Tensor, scale: int) -> torch.Tensor:
    """
    Downscale a flow field by `scale` using average pooling,
    and divide flow values by `scale` so they remain in the new coordinate space.

    Args:
        flow  : (B, 2, H, W)
        scale : int downsampling factor (e.g. 8)

    Returns:
        (B, 2, H//scale, W//scale)
    """
    downsampled = F.avg_pool2d(flow, kernel_size=scale, stride=scale)
    return downsampled / scale


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    PSNR in dB.  Inputs assumed in [0, 1], shape (B, C, H, W) or (C, H, W).
    """
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float("inf")
    return (-10.0 * torch.log10(mse)).item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    from skimage.metrics import structural_similarity as sk_ssim
    # Handle (B, T, C, H, W) by flattening to (B*T, C, H, W)
    if pred.ndim == 5:
        B, T, C, H, W = pred.shape
        pred = pred.reshape(B * T, C, H, W)
        target = target.reshape(B * T, C, H, W)
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    scores = []
    for p, t in zip(pred_np, target_np):
        p_hw = p.transpose(1, 2, 0)
        t_hw = t.transpose(1, 2, 0)
        s = sk_ssim(p_hw, t_hw, data_range=1.0, channel_axis=-1)
        scores.append(s)
    return float(np.mean(scores))

_lpips_model = None


def compute_lpips(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Mean LPIPS (AlexNet) over batch.  Inputs in [0, 1], shape (B, C, H, W).
    Model is loaded once and cached.
    """
    global _lpips_model
    if _lpips_model is None:
        import lpips
        _lpips_model = lpips.LPIPS(net="alex")
        _lpips_model.eval()
    _lpips_model = _lpips_model.to(pred.device)
    # lpips expects inputs in [-1, 1]
    with torch.no_grad():
        score = _lpips_model(pred * 2 - 1, target * 2 - 1)
    return score.mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Loss helpers
# ─────────────────────────────────────────────────────────────────────────────

def focal_loss(pred: torch.Tensor, target: torch.Tensor,
               gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
    """
    Sigmoid focal loss for binary classification.

    Args:
        pred   : (B,) or (B, 1) — raw logits or probabilities
        target : (B,) or (B, 1) — binary labels {0, 1}
    """
    pred = pred.view(-1)
    target = target.view(-1).float()
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    p_t = torch.exp(-bce)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    loss = alpha_t * (1 - p_t) ** gamma * bce
    return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer = None,
                    device: torch.device = torch.device("cpu")) -> int:
    """
    Load model (and optionally optimizer) state from checkpoint.

    Returns:
        start_epoch (int): epoch to resume from
    """
    ckpt = torch.load(path, map_location=device)
    if "model" in ckpt: 
        model.load_state_dict(ckpt["model"])
    elif "temporal" in ckpt: 
        model.load_state_dict(ckpt["temporal"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0)


# ─────────────────────────────────────────────────────────────────────────────
# RLE mask decoder  (CLEVRER derender_proposals format)
# ─────────────────────────────────────────────────────────────────────────────

def decode_rle_mask(rle: dict) -> np.ndarray:
    """
    Decode a COCO-style RLE mask stored as {'size': [H, W], 'counts': <string>}.

    Returns:
        mask: np.ndarray bool of shape (H, W)
    """
    h, w = rle["size"]
    counts_str = rle["counts"]

    # The counts may be a plain int-list string or a COCO binary RLE string.
    # CLEVRER uses the compressed binary RLE (bytes encoded as a string).
    try:
        # Try pycocotools first (fastest, most correct)
        from pycocotools import mask as coco_mask
        mask = coco_mask.decode(rle).astype(bool)
        return mask
    except ImportError:
        pass

    # Fallback: manual COCO binary RLE decoding
    # counts encodes alternating 0-run and 1-run lengths, column-major order
    if isinstance(counts_str, list):
        # Uncompressed RLE
        counts = counts_str
    else:
        # Compressed binary RLE: decode byte string
        counts = _decode_coco_rle_string(counts_str)

    total = h * w
    flat = np.zeros(total, dtype=bool)
    idx = 0
    val = False
    for c in counts:
        flat[idx: idx + c] = val
        idx += c
        val = not val

    # COCO RLE is column-major (Fortran order)
    return flat.reshape((h, w), order="F")


def _decode_coco_rle_string(s: str) -> list:
    """Decode a COCO compressed RLE byte string to run-length counts."""
    counts = []
    p = 0
    while p < len(s):
        x = 0
        k = 0
        more = True
        while more:
            c = ord(s[p]) - 48
            p += 1
            more = (c & 32) != 0
            x |= (c & 31) << (5 * k)
            k += 1
        if len(counts) > 2 and x <= counts[-2]:
            x += counts[-2]
        counts.append(x)
    return counts
