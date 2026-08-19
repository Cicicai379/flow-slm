"""Blockwise flow-matching research components.

This package is intentionally isolated from the root-level block prototype.
Code moves into the training path only after its invariants are covered here.
"""

from .blocks import BlockBatch, pack_blocks, shift_blocks_right

__all__ = ["BlockBatch", "pack_blocks", "shift_blocks_right"]
