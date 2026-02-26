"""
models.py — Core neural network modules for CLEVRER video prediction.

Modules
-------
VAEEncoder       : (3, H, W) → mu (C_z, H_z, W_z) + log_var (C_z, H_z, W_z)
VAEDecoder       : (C_z, H_z, W_z) → (3, H, W)
VAE              : encoder + decoder + reparameterise

ConvLSTMCell     : single ConvLSTM cell (spatial hidden state)
TemporalModel    : ConvLSTM backbone + FlowHead + OcclusionHead +
                   ResidualHead + StateHead + CollisionHead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import backward_warp


# ─────────────────────────────────────────────────────────────────────────────
# VAE
# ─────────────────────────────────────────────────────────────────────────────

class VAEEncoder(nn.Module):
    def __init__(self, c_z: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),   # H/2, W/2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # H/4, W/4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 2 * c_z, kernel_size=4, stride=2, padding=1),  # H/8, W/8
        )
        self.c_z = c_z

    def forward(self, x: torch.Tensor):
        """
        x : (B, 3, H, W)
        Returns mu, log_var each (B, C_z, H/8, W/8)
        """
        out = self.net(x)
        mu, log_var = out.chunk(2, dim=1)
        return mu, log_var


class VAEDecoder(nn.Module):
    def __init__(self, c_z: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(c_z, 64, kernel_size=4, stride=2, padding=1),  # H/4, W/4
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # H/2, W/2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),    # H, W
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VAE(nn.Module):
    def __init__(self, c_z: int = 32):
        super().__init__()
        self.encoder = VAEEncoder(c_z)
        self.decoder = VAEDecoder(c_z)
        self.c_z = c_z

    def reparameterise(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def encode(self, x: torch.Tensor):
        mu, log_var = self.encoder(x)
        z = self.reparameterise(mu, log_var)
        return z, mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        z, mu, log_var = self.encode(x)
        recon = self.decode(z)
        return recon, mu, log_var

    @staticmethod
    def kl_loss(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())


# ─────────────────────────────────────────────────────────────────────────────
# ConvLSTM
# ─────────────────────────────────────────────────────────────────────────────

class ConvLSTMCell(nn.Module):
    """
    Single ConvLSTM cell.  Maintains a spatial hidden state (C_lstm, H_z, W_z).

    Gates use 3×3 convolutions applied to the concatenation of input and hidden.
    """

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,   # i, f, g, o gates
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x: torch.Tensor,
                state: tuple[torch.Tensor, torch.Tensor]):
        """
        x     : (B, C_in, H, W)
        state : (h, c) each (B, C_lstm, H, W)
        Returns new (h, c).
        """
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

    def init_state(self, batch_size: int, h: int, w: int,
                   device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros(batch_size, self.hidden_channels, h, w, device=device)
        return zeros, zeros.clone()


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Model heads
# ─────────────────────────────────────────────────────────────────────────────

class FlowHead(nn.Module):
    """
    Predicts optical flow in *latent* space: (2, H_z, W_z).
    Output is tanh-scaled to [-max_disp, max_disp].
    """

    def __init__(self, c_lstm: int, max_disp: float = 1.0):
        super().__init__()
        self.max_disp = max_disp
        self.net = nn.Sequential(
            nn.Conv2d(c_lstm, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=3, padding=1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(h)) * self.max_disp


class OcclusionHead(nn.Module):
    """
    Predicts soft occlusion mask in latent space: (1, H_z, W_z).
    Value 1 = trustworthy (no occlusion), 0 = occluded region.
    """

    def __init__(self, c_lstm: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_lstm, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class ResidualHead(nn.Module):
    """
    Corrects the warped latent.  Takes warped_z as input, outputs an additive
    residual of the same shape (C_z, H_z, W_z).
    """

    def __init__(self, c_z: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_z, c_z * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_z * 2, c_z, kernel_size=3, padding=1),
        )

    def forward(self, warped_z: torch.Tensor) -> torch.Tensor:
        return self.net(warped_z)


class StateHead(nn.Module):
    """
    Auxiliary head: predicts per-object state (location + velocity) from the
    global average of the ConvLSTM hidden state.

    Output: (B, N_max, state_dim) — state_dim = 6 (x,y,z, vx,vy,vz)
    """

    def __init__(self, c_lstm: int, n_max: int = 6, state_dim: int = 6):
        super().__init__()
        self.n_max = n_max
        self.state_dim = state_dim
        self.net = nn.Sequential(
            nn.Linear(c_lstm, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_max * state_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, C_lstm, H_z, W_z)"""
        pooled = h.mean(dim=(-2, -1))          # (B, C_lstm)
        out = self.net(pooled)                  # (B, N_max * state_dim)
        return out.view(-1, self.n_max, self.state_dim)


class CollisionHead(nn.Module):
    """
    Predicts collision probability (scalar per frame) using focal loss supervision.
    Returns raw logits (apply sigmoid for probability).
    """

    def __init__(self, c_lstm: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(c_lstm, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, C_lstm, H_z, W_z) → (B,) logits"""
        pooled = h.mean(dim=(-2, -1))     # (B, C_lstm)
        return self.net(pooled).squeeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# TemporalModel
# ─────────────────────────────────────────────────────────────────────────────

class TemporalModel(nn.Module):
    """
    Full temporal prediction model.

    At each step t it:
      1. Projects z_t to C_lstm channels via a 1×1 conv.
      2. Runs a ConvLSTM step to update (h_t, c_t).
      3. Predicts flow_z, occlusion mask, collision logit, object states.
      4. Warps z_t using flow_z → warped_z.
      5. ResidualHead(warped_z) → residual.
      6. final_z = warped_z * mask + residual * (1 - mask).
      7. Optionally decodes final_z to a predicted frame (when vae is provided).
    """

    def __init__(self, c_z: int = 32, c_lstm: int = 64,
                 max_disp: float = 1.0, n_max: int = 6, state_dim: int = 6):
        super().__init__()
        self.c_z = c_z
        self.c_lstm = c_lstm

        self.input_proj = nn.Conv2d(c_z, c_lstm, kernel_size=1)
        self.conv_lstm = ConvLSTMCell(c_lstm, c_lstm)

        self.flow_head = FlowHead(c_lstm, max_disp)
        self.occ_head = OcclusionHead(c_lstm)
        self.residual_head = ResidualHead(c_z)
        self.state_head = StateHead(c_lstm, n_max, state_dim)
        self.collision_head = CollisionHead(c_lstm)

    def init_state(self, batch_size: int, h_z: int, w_z: int,
                   device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return self.conv_lstm.init_state(batch_size, h_z, w_z, device)

    def step(self, z_t: torch.Tensor,
             state: tuple[torch.Tensor, torch.Tensor],
             vae_decoder: nn.Module = None):
        """
        Single forward step.

        Args:
            z_t        : (B, C_z, H_z, W_z)
            state      : (h, c) from previous step
            vae_decoder: if provided, decodes final_z to pixel space

        Returns dict with keys:
            flow_z, mask, warped_z, residual, final_z,
            collision_logit, state_pred, predicted_frame (if vae_decoder given),
            new_state
        """
        x = self.input_proj(z_t)
        h, c = self.conv_lstm(x, state)

        flow_z = self.flow_head(h)
        mask = self.occ_head(h)
        collision_logit = self.collision_head(h)
        state_pred = self.state_head(h)

        warped_z = backward_warp(z_t, flow_z)
        residual = self.residual_head(warped_z)
        final_z = warped_z * mask + residual * (1.0 - mask)

        out = {
            "flow_z": flow_z,
            "mask": mask,
            "warped_z": warped_z,
            "residual": residual,
            "final_z": final_z,
            "collision_logit": collision_logit,
            "state_pred": state_pred,
            "new_state": (h, c),
        }

        if vae_decoder is not None:
            out["predicted_frame"] = vae_decoder(final_z)

        return out

    def forward(self, z_seq: torch.Tensor, state=None, vae_decoder=None):
        """
        Unroll over a sequence of latents.

        Args:
            z_seq      : (B, T, C_z, H_z, W_z)
            state      : initial (h, c); if None, zeros are used
            vae_decoder: optional VAEDecoder for pixel-space output

        Returns:
            outputs : list of T-1 step dicts (step from t=0 predicts t=1, etc.)
            final_state : (h, c) after last step
        """
        B, T, C_z, H_z, W_z = z_seq.shape
        if state is None:
            state = self.init_state(B, H_z, W_z, z_seq.device)

        outputs = []
        for t in range(T - 1):
            out = self.step(z_seq[:, t], state, vae_decoder)
            state = out["new_state"]
            outputs.append(out)

        return outputs, state
