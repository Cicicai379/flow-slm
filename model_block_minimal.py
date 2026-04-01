from typing import Optional, Tuple, Union, List
import math
import logging

import torch
import torch.nn as nn
from contextlib import nullcontext

logger = logging.getLogger(__name__)

def modulate(x, shift, scale):
    """Apply modulation to input tensor."""
    return x * (1 + scale) + shift

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000, scale: float = 1000.0) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=t.dtype) / half
        ).to(device=t.device)
        args = t[:, :, None].float() * scale * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Forward pass for timestep embedding."""
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

class ResBlock(nn.Module):
    """A residual block with adaptive layer normalization."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass for residual block with adaptive layer norm."""
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h

class FinalLayer(nn.Module):
    """The final layer adopted from DiT."""

    def __init__(self, model_channels: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass for final layer."""
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class BlockFlowNet(nn.Module):
    """Flow matching network for joint block denoising with adaptive layer normalization."""

    def __init__(
        self,
        block_dim: int,  # block_size * feature_dim (flattened block dimension)
        model_channels: int,
        z_channels: int,  # conditioning dimension from LM
        num_res_blocks: int,
        grad_checkpointing: bool = False
    ):
        super().__init__()

        self.block_dim = block_dim
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        # Project flattened noisy block to model dimension
        self.input_proj = nn.Linear(block_dim, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(model_channels))

        self.res_blocks = nn.ModuleList(res_blocks)

        # Output back to flattened block space for joint prediction
        self.final_layer = FinalLayer(model_channels, block_dim)

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize model weights."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Apply block flow network to noisy flattened block.

        Args:
            x: [B, block_dim] flattened noisy block
            t: [B] timestep for each batch element
            c: [B, z_channels] conditioning from LM for this block

        Returns:
            [B, block_dim] predicted velocity/noise for joint block
        """
        x = self.input_proj(x)  # [B, block_dim] → [B, model_channels]
        t = self.time_embed(t.unsqueeze(1))  # [B] → [B, 1, model_channels]
        t = t.squeeze(1)  # [B, model_channels]
        c = self.cond_embed(c)  # [B, z_channels] → [B, model_channels]

        y = t + c  # [B, model_channels]

        if self.grad_checkpointing and not torch.jit.is_scripting():
            from torch.utils.checkpoint import checkpoint
            for block in self.res_blocks:
                x = checkpoint(block, x, y.unsqueeze(1))
                x = x.squeeze(1)
        else:
            for block in self.res_blocks:
                x = block(x.unsqueeze(1), y.unsqueeze(1))  # Add sequence dim for ResBlock
                x = x.squeeze(1)  # Remove sequence dim

        out = self.final_layer(x.unsqueeze(1), y.unsqueeze(1))  # Add sequence dim
        out = out.squeeze(1)  # [B, block_dim]
        return out