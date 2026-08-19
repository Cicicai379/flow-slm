# Blockwise Flow-SLM

This directory is the clean implementation track for blockwise flow matching
and decoding. The existing root-level `*_block.py` files are treated as a
prototype/reference and are not modified by this scaffold.

## Objective

Generate `K` consecutive Mimi latent frames with one causal-LM call and one
joint flow solve. At 12.5 latent frames/second, block size `K=8` reduces the
number of autoregressive decoder iterations by up to 8x; actual end-to-end
speedup must include the larger flow head and ODE solver cost.

The causal contract is:

```text
LM input:      [BOS, block 0, block 1, ..., block N-2]
flow targets:  [block 0, block 1, block 2, ..., block N-1]
```

No representation used to predict block `i` may depend on block `i`. Partial
last blocks carry a frame-level mask; zero padding never contributes to loss.

## Layout

- `PLAN.md`: phased implementation and go/no-go gates.
- `blocks.py`: block packing, masks, and causal shifting.
- `experiment_matrix.yaml`: controlled quality/speed ablations.
- `tests/`: CPU-fast invariant tests.

Run the current checks from the repository root:

```bash
python -m unittest blockwise.tests.test_blocks
```

All future checkpoints, caches, generated audio, and W&B buffers must live
under node-local `/data/cicicai/flow_slm/blockwise`. Only code, compact logs,
and final result tables belong on `/accounts`.

The current full Block-8 run uses two H200s, batch size 4 per GPU, and 16-step
gradient accumulation for effective global batch size 128.
