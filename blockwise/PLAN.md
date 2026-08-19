# Blockwise flow matching and decoding plan

## Success criteria

The main comparison is against the reproduced framewise 270M checkpoint using
the same prompts, Mimi representation, ODE solver, and evaluation stack.

1. At least 4x fewer causal-decoder calls for block size 8.
2. At least 2x end-to-end generation speedup at matched batch size.
3. No more than 1 absolute point regression on each SALMon category.
4. No material intelligibility regression on the paper's 500-prompt mixture.
5. Stable training: finite gradients and validation loss for 10k debug steps.

## Phase 0 — Freeze semantics and baseline

- Record framewise checkpoint, commit, config, seeds, prompt list, and W&B run.
- Benchmark decoder calls, ODE evaluations, real-time factor, and peak memory.
- Define one block as `K` consecutive normalized Mimi frames after feature
  reduction; start with reduction factor 1.
- Use encoded frame counts everywhere. Convert waveform lengths exactly once.

Gate: the baseline generation and metrics are reproducible from one manifest.

## Phase 1 — Correct block construction

- Use `pack_blocks` to return values, block masks, and within-block frame masks.
- Right-shift teacher-forcing inputs with a learned BOS block.
- Apply frame masks before flow-loss reduction, including partial last blocks.
- Vectorize valid blocks as `[B*N, K, D]`; avoid a Python loop over blocks.
- Add tests for empty/short sequences, exact boundaries, mixed lengths, and
  reduction-factor interactions.

Gate: target-leakage test, mask tests, shape tests, and finite backward pass.

## Phase 2 — Joint flow head

- Implement two interchangeable heads:
  - flattened-block MLP as the simplest controlled baseline;
  - intra-block Transformer with positional embeddings for temporal structure.
- Draw one flow time per block initially; separately ablate one time per frame.
- Preserve the original flow path, logit-normal time distribution, sigma floor,
  and classifier-free conditioning dropout.
- Train block size 1 and require numerical/metric parity with framewise FM.

Gate: block size 1 matches the original loss on an identical synthetic batch.

## Phase 3 — Autoregressive block decoding

- Cache causal-LM key/value states so each generated block adds one LM step.
- Generate `[B, K, D]` jointly with Euler first; add higher-order solvers only
  after parity.
- Define prompt-boundary behavior explicitly: preserve a partial prompt block
  and generate only its missing frames, rather than silently inserting zeros.
- Predict EOS/length at block level plus an intra-block stop offset in
  `[1, K]`; mask decoded frames after the stop.
- Keep CFG batching and conditional/unconditional cache handling deterministic.

Gate: fixed-seed decoding is deterministic and never returns padded audio.

## Phase 4 — Token auxiliary objective

The prototype repeats one block-level token distribution across all frames,
which cannot express distinct within-block tokens. Replace it with either:

- `K` offset-specific token heads per block, or
- a boundary-only semantic-token objective documented as such.

Start without the auxiliary objective to isolate block flow behavior, then add
the offset-specific version and measure its effect.

Gate: targets/logits/masks have identical valid-token counts and EOS alignment.

## Phase 5 — Training ladder

1. CPU unit tests and synthetic backward pass.
2. One H200, 100 steps, tiny local audio subset.
3. One H200, 10k steps, MLS-10k; compare K=1, 4, 8.
4. Four H200s, full MLS only after K=1 parity and K=8 quality are established.

Every GPU run logs code/config identity, block size, effective frames per batch,
throughput, memory, train/valid losses, and sample audio to W&B. Launch through
`safe`; use only `/data/cicicai/flow_slm/blockwise` for high-volume artifacts.

## Phase 6 — Evaluation and decision

- Run the exact framewise and blockwise prompt sets with matched decoding knobs.
- Report quality versus real-time factor and decoder-call reduction, not quality
  alone.
- Run three seeds for shortlisted K values and report mean plus standard error.
- Promote the smallest K meeting quality gates; do not assume K=8 is optimal.

## Prototype audit items to resolve during migration

- Replace implicit previous-index conditioning with an explicit shifted input.
- Mask frames inside partial blocks, not only whole blocks.
- Remove the per-block Python loss loop.
- Fix partial-prompt handling at generation boundaries.
- Give auxiliary token logits within-block positional capacity.
- Restore the paper-era 16-to-24 kHz resampling contract when using MLS.
- Add online W&B logging and stable run IDs before cluster experiments.
