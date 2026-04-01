import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from model_block_minimal import BlockFlowNet

class BlockFlowLoss(nn.Module):
    """Block flow matching loss for joint block modeling.

    Each block is treated as a single high-dimensional variable for flow matching.
    """
    def __init__(
        self,
        block_dim: int,         # Flattened block dimension (block_size * feature_dim)
        z_dim: int,             # Conditioning dimension from LM
        sigma_min: float = 1e-5,
        t_dist: str = "uniform",
        null_prob: float = 0.05,
        model_channels: int = 512,
        num_res_blocks: int = 3,
    ):
        super().__init__()
        self.block_dim = block_dim
        self.z_dim = z_dim
        self.sigma_min = sigma_min
        self.t_dist = t_dist
        self.null_prob = null_prob

        # Null conditioning embedding for classifier-free guidance
        self.null_emb = nn.Embedding(1, z_dim)

        # Block flow network for joint denoising
        self.net = BlockFlowNet(
            block_dim=block_dim,
            model_channels=model_channels,
            z_channels=z_dim,
            num_res_blocks=num_res_blocks,
        )

    def forward(self, z: torch.Tensor, target_block: torch.Tensor) -> torch.Tensor:
        """Compute flow loss for a single block.

        Args:
            z: [B, z_dim] LM conditioning for this block
            target_block: [B, block_dim] flattened target block

        Returns:
            loss: [B, block_dim] per-element block loss
        """
        batch_size = target_block.shape[0]
        device = target_block.device
        dtype = target_block.dtype

        # Sample timestep for each batch element
        if self.t_dist == "uniform":
            t = torch.rand([batch_size], device=device, dtype=dtype)
        elif self.t_dist == "logit_normal":
            t = torch.sigmoid(torch.randn([batch_size], device=device, dtype=dtype))
        else:
            raise NotImplementedError(f"t_dist {self.t_dist} not implemented")

        # Add noise to entire flattened block (joint modeling)
        noise = torch.randn_like(target_block)  # [B, block_dim]
        t_expand = t.unsqueeze(1)  # [B, 1]

        # Flow matching parameterization
        psi_t = (1 - (1 - self.sigma_min) * t_expand) * noise + t_expand * target_block
        u = target_block - (1 - self.sigma_min) * noise

        # Apply classifier-free guidance during training
        if self.training:
            sample_null = torch.rand(batch_size, device=device, dtype=dtype)
            is_null = (sample_null < self.null_prob).float()  # [B]
            z = z * (1 - is_null).unsqueeze(1) + self.null_emb.weight * is_null.unsqueeze(1)

        # Predict velocity/noise for the joint block
        out = self.net(psi_t, t, z)  # [B, block_dim]

        # MSE loss over the entire flattened block
        loss = F.mse_loss(out, u, reduction='none')  # [B, block_dim]

        return loss