"""Shape-safe block packing and causal teacher-forcing utilities."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BlockBatch:
    """A padded batch of acoustic frames grouped into fixed-size blocks."""

    values: torch.Tensor
    frame_mask: torch.Tensor
    block_mask: torch.Tensor
    frame_lengths: torch.Tensor

    @property
    def flattened(self) -> torch.Tensor:
        batch, blocks, block_size, feature_dim = self.values.shape
        return self.values.reshape(batch, blocks, block_size * feature_dim)


def pack_blocks(
    frames: torch.Tensor,
    frame_lengths: torch.Tensor,
    block_size: int,
    pad_value: float = 0.0,
) -> BlockBatch:
    """Pack ``[B, T, D]`` frames without losing partial-block mask detail.

    ``frame_lengths`` is expressed in encoded acoustic frames, not waveform
    samples or normalized duration. Invalid frames never contribute to loss.
    """

    if frames.ndim != 3:
        raise ValueError(f"frames must have shape [B, T, D], got {tuple(frames.shape)}")
    if frame_lengths.ndim != 1 or frame_lengths.shape[0] != frames.shape[0]:
        raise ValueError("frame_lengths must have shape [B]")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    batch, time, feature_dim = frames.shape
    lengths = frame_lengths.to(device=frames.device, dtype=torch.long)
    if torch.any(lengths < 0) or torch.any(lengths > time):
        raise ValueError(f"frame_lengths must lie in [0, {time}]")

    num_blocks = max(1, (time + block_size - 1) // block_size)
    padded_time = num_blocks * block_size
    if padded_time != time:
        frames = F.pad(frames, (0, 0, 0, padded_time - time), value=pad_value)

    values = frames.reshape(batch, num_blocks, block_size, feature_dim)
    flat_positions = torch.arange(padded_time, device=frames.device)
    frame_mask = (flat_positions.unsqueeze(0) < lengths.unsqueeze(1)).reshape(
        batch, num_blocks, block_size
    )
    block_mask = frame_mask.any(dim=-1)
    return BlockBatch(values, frame_mask, block_mask, lengths)


def shift_blocks_right(targets: torch.Tensor, bos_block: torch.Tensor) -> torch.Tensor:
    """Create causal LM inputs ``[BOS, target_0, ..., target_(N-2)]``.

    The returned position ``i`` can be used to condition flow matching for
    target block ``i`` without exposing that target to the causal decoder.
    """

    if targets.ndim != 3:
        raise ValueError(f"targets must have shape [B, N, block_dim], got {tuple(targets.shape)}")
    if bos_block.ndim != 1 or bos_block.shape[0] != targets.shape[-1]:
        raise ValueError("bos_block must have shape [block_dim]")

    bos = bos_block.to(device=targets.device, dtype=targets.dtype)
    bos = bos.reshape(1, 1, -1).expand(targets.shape[0], 1, -1)
    return torch.cat((bos, targets[:, :-1]), dim=1)
