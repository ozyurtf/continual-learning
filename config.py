import argparse
import torch


def get_device(req: str = "auto") -> torch.device:
    if req != "auto":
        return torch.device(req)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_args():
    p = argparse.ArgumentParser(description="CLEVRER Video Prediction")

    # ── Data ──────────────────────────────────────────────────────────────────
    p.add_argument("--data_root", default=".",
                   help="Root directory containing video_train/, annotation_train/, etc.")
    p.add_argument("--flow_dir", default="precomputed",
                   help="Directory for precomputed flow/mask/physics-flow arrays")
    p.add_argument("--checkpoint_dir", default="checkpoints",
                   help="Directory for saved checkpoints")
    p.add_argument("--video_folders", default=None,
                   help="Comma-separated subfolder names to restrict training/inference "
                        "(e.g. 'video_00000-01000' for local testing). None = all folders.")
    p.add_argument("--max_videos", type=int, default=None,
                   help="Cap the total number of videos used (e.g. 5 for quick local tests). "
                        "Applied after folder filtering.")

    # ── Image / latent dimensions ─────────────────────────────────────────────
    p.add_argument("--img_h", type=int, default=320, help="Frame height after resize")
    p.add_argument("--img_w", type=int, default=480, help="Frame width after resize")
    p.add_argument("--c_z", type=int, default=32, help="VAE latent channels")
    # Latent spatial dims are always img_h/8 × img_w/8 (3 stride-2 conv layers)

    # ── Temporal model ────────────────────────────────────────────────────────
    p.add_argument("--c_lstm", type=int, default=64,
                   help="ConvLSTM hidden channels")
    p.add_argument("--seq_len", type=int, default=10,
                   help="Training sequence length (frames unrolled through ConvLSTM)")
    p.add_argument("--max_disp", type=float, default=1.0,
                   help="Max flow displacement in latent space (tanh scale)")
    p.add_argument("--n_max_objects", type=int, default=6,
                   help="Max number of objects per scene for StateHead")

    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument("--phase", choices=["vae", "temporal", "joint"],
                   default="vae", help="Training phase")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta_kl", type=float, default=1e-4,
                   help="KL divergence weight in VAE loss")
    p.add_argument("--unfreeze_vae", action="store_true",
                   help="Also update VAE weights during temporal/joint phase")
    p.add_argument("--vae_lr_scale", type=float, default=0.01,
                   help="LR multiplier for VAE in joint phase")

    # Loss weights (temporal phase)
    p.add_argument("--w_latent", type=float, default=1.0)
    p.add_argument("--w_flow", type=float, default=1.0)
    p.add_argument("--w_physics_flow", type=float, default=0.3)
    p.add_argument("--w_occ", type=float, default=1.0)
    p.add_argument("--w_warp", type=float, default=1.0)
    p.add_argument("--w_residual", type=float, default=1.0)
    p.add_argument("--w_state", type=float, default=0.1)
    p.add_argument("--w_collision", type=float, default=0.5)

    # ── Checkpoints ───────────────────────────────────────────────────────────
    p.add_argument("--checkpoint", default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--vae_checkpoint", default=None,
                   help="Path to VAE checkpoint (used in temporal/joint/inference phases)")
    p.add_argument("--temporal_checkpoint", default=None,
                   help="Path to temporal model checkpoint (used in inference phase)")
    p.add_argument("--save_every", type=int, default=5,
                   help="Save checkpoint every N epochs")

    # ── DataLoader ────────────────────────────────────────────────────────────
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers (0 recommended on MPS/Mac)")

    # ── Inference ─────────────────────────────────────────────────────────────
    p.add_argument("--num_input_frames", type=int, default=5,
                   help="Real frames fed to LSTM before autoregressive prediction")
    p.add_argument("--num_pred_frames", type=int, default=10,
                   help="Number of frames to predict autoregressively")
    p.add_argument("--ttt", action="store_true",
                   help="Enable Test-Time Training (fine-tunes heads per video)")
    p.add_argument("--ttt_steps", type=int, default=10,
                   help="Number of TTT gradient steps per video")
    p.add_argument("--ttt_lr", type=float, default=1e-4)
    p.add_argument("--output_dir", default="outputs",
                   help="Directory to save predicted MP4s and metrics")

    # ── Precompute ────────────────────────────────────────────────────────────
    p.add_argument("--split", choices=["train", "val", "test"], default="train",
                   help="Dataset split to precompute flows for")
    p.add_argument("--fb_alpha", type=float, default=0.5,
                   help="Alpha parameter for forward-backward consistency mask")

    # ── Device ────────────────────────────────────────────────────────────────
    p.add_argument("--device", default="cpu",
                   help="Device: 'auto' (cuda→mps→cpu), 'cuda', 'mps', or 'cpu'")

    args = p.parse_args()
    args.device = get_device(args.device)

    # Derived dimensions
    args.h_z = args.img_h // 8
    args.w_z = args.img_w // 8

    # Parse video_folders into a list
    if args.video_folders is not None:
        args.video_folders = [f.strip() for f in args.video_folders.split(",")]

    return args
