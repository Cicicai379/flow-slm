# Flow-SLM Reproduction and Blockwise Flow-Matching Report

**Report date:** 2026-08-20

**Repository:** <https://github.com/Cicicai379/flow-slm>

**Evaluation-code snapshot:** [`20384c2`](https://github.com/Cicicai379/flow-slm/commit/20384c2)

**Versioned audio snapshot:** [`79557c4`](https://github.com/Cicicai379/flow-slm/commit/79557c4)

**Operator / Slurm user:** `cicicai`

## Executive summary

Two models were trained on the full MLS-English training split:

1. **Original Flow-SLM 270M reproduction.** This was trained by us from step 0
   to the configured 85,000 optimizer steps. It did **not** initialize from the
   authors' released trained Flow-SLM checkpoint. As in the original
   architecture, it used the standard pretrained Mimi encoder and pretrained
   OpenELM-270M backbone as components.
2. **Block-8 Flow-SLM.** This is our separate blockwise flow-matching variant.
   It was trained to 85,000 optimizer steps and predicts eight acoustic frames
   jointly per flow-matching target.

Both production jobs completed successfully with exit code `0:0`, wrote final
DeepSpeed checkpoints, and logged online to W&B. The original training protocol
has therefore been reproduced operationally. We also exported and strictly
loaded both final checkpoints, completed the public likelihood-based downstream
suite (sWUGGY, sBLIMP, and SALMon), and generated matched audio continuations.

Our self-trained original model exceeded the paper's sWUGGY, sBLIMP, and SALMon
consistency values, but missed SALMon sentiment alignment by 5.50 points and
background alignment by 1.00 point. It is therefore an operational training
reproduction with strong downstream results, **not an exact reproduction of
every paper metric**. The repository does not include the paper's complete
genPPL, speaker-similarity, emotion2vec-FSD, or WavLM-FSD evaluator pipelines;
those four generation scores remain unexecuted rather than estimated.

An earlier SALMon evaluation used the **authors' released checkpoint only** as
a compatibility/evaluator baseline. Those scores must not be attributed to our
newly trained checkpoint.

## Provenance: our training versus the released checkpoint

### Our original training run

- Slurm job: `3456432` (`flowslm-paper-270m`)
- Started at step 0; the log begins with losses near random initialization:
  flow loss about `1.01`, token loss about `33.1`, and token accuracy near zero.
- The log contains no `loading from`, `Restoring states`, or explicit
  `--ckpt_path` load before training.
- The production launcher does not pass the authors' `270m.bin` checkpoint.
- Training reached `max_steps=85000` and exited normally.

The model is not trained with every parameter initialized randomly: Mimi and
OpenELM are loaded from their standard pretrained releases, which is part of
the Flow-SLM architecture. The important distinction is that no already-trained
Flow-SLM checkpoint from the paper was used to initialize this run.

### Released checkpoint use

The released checkpoint at
`/data/cicicai/flow_slm/paper_reproduction/checkpoints/270m.bin` was used only
for:

- strict compatibility/forward-pass testing in job `3453415`; and
- the standalone SALMon evaluator baseline in jobs `3453501` and `3453584`.

It was not used by production training job `3456432`.

## Data

| Item | Value |
|---|---:|
| Dataset | `parler-tts/mls_eng`, default configuration |
| Training rows | 10,808,037 |
| Development rows | 3,807 |
| Paper-reported base hours | 44,659.74 |
| Effective input sample rate | 16 kHz |
| Mimi model sample rate | 24 kHz, using model-side resampling |
| Lorax Arrow cache size | 659 GiB (`du`) |
| Verified cache location | `/data/cicicai/flow_slm/paper_reproduction/hf_datasets` |

The Arrow cache was copied from Thidwick to Lorax by CPU-only job `3456006`.
The copy completed on 2026-08-16 at 16:11:29 PDT, after which both splits were
loaded in offline mode and their row counts were verified. High-volume data,
checkpoints, extension caches, and W&B buffers stayed under node-local
`/data/cicicai`; `/accounts` contains only code, logs, and reporting artifacts.

## Training configurations

### Original Flow-SLM 270M

| Setting | Effective value |
|---|---:|
| Decoder | OpenELM-270M |
| Acoustic encoder | Mimi, frozen |
| Flow objective | framewise flow matching (`FM`) |
| Semantic token loss | enabled, weight 1.0 |
| Future semantic tokens | 4 |
| Optimizer | AdamW8bit |
| Learning rate | 5e-4 |
| Warmup | 5,000 optimizer steps |
| Weight decay | 0.01 |
| CFG/null-conditioning dropout | 0.05 |
| Per-GPU batch size | 32 |
| GPUs | 4 H200s on one Lorax node |
| Gradient accumulation | 1 |
| Global batch size | 128 |
| Precision | BF16 mixed |
| Distributed strategy | DeepSpeed ZeRO Stage 3 |
| Maximum optimizer steps | 85,000 |

Effective configuration snapshot:
`/data/cicicai/flow_slm/paper_reproduction/training/flow-slm-270m-mls/270m/conf.yaml`.

### Block-8 Flow-SLM

| Setting | Effective value |
|---|---:|
| Decoder | OpenELM-270M |
| Acoustic encoder | Mimi, frozen |
| Flow objective | block flow matching (`BLOCK_FM`) |
| Block size | 8 acoustic frames |
| Semantic token loss | disabled |
| Optimizer | AdamW8bit |
| Learning rate | 5e-4 |
| Warmup | 5,000 optimizer steps |
| Weight decay | 0.01 |
| CFG/null-conditioning dropout | 0.05 |
| Per-GPU batch size | 4 |
| GPUs | 2 H200s on one Lorax node |
| Gradient accumulation | 16 |
| Global batch size | 128 |
| Precision | BF16 mixed |
| Distributed strategy | DeepSpeed ZeRO Stage 2 |
| Maximum optimizer steps | 85,000 |

Effective configuration snapshot:
`/data/cicicai/flow_slm/blockwise/runs/270m-block8-mls-v1/270m_block8_mls/conf.yaml`.

## Production training results

| Model | Job | State | Runtime | Final step | Final reported train metrics | Final reported validation metrics |
|---|---:|---|---:|---:|---|---|
| Original Flow-SLM 270M | `3456432` | Completed (`0:0`) | 11:57:25 | 85,000 | flow loss `0.00761`; token loss `3.39`; token accuracy `0.527` | flow loss `0.00723`; token loss `4.18`; token accuracy `0.421` |
| Block-8 Flow-SLM | `3456847` | Completed (`0:0`) | 2 days 03:22:26 | 85,000 | final displayed flow loss `0.0251` (recent values roughly `0.019`-`0.029`) | flow loss `0.0244` |

These are optimization and held-out development losses. They are not directly
comparable to the paper's downstream spoken-language and generation metrics.

### W&B runs

- Original: <https://wandb.ai/cicicai-university-of-california-berkeley/flow-slm-paper-reproduction/runs/flowslm-270m-mls-paper>
- Block-8: <https://wandb.ai/cicicai-university-of-california-berkeley/flow-slm-blockwise/runs/flowslm-270m-block8-mls-v1>

Both production launchers verify W&B authentication before spending GPU time,
use stable run IDs, and buffer W&B state on node-local storage.

## Checkpoints

### Original final checkpoint

```text
/data/cicicai/flow_slm/paper_reproduction/training/flow-slm-270m-mls/270m/model-step=0085000.ckpt
```

- DeepSpeed checkpoint size: approximately 3.1 GiB.
- `last.ckpt` is also current internally and approximately 3.1 GiB.
- Retained step checkpoints: 65k, 70k, 75k, 80k, and 85k.

### Block-8 final checkpoint

```text
/data/cicicai/flow_slm/blockwise/runs/270m-block8-mls-v1/270m_block8_mls/model-step=0085000.ckpt
```

- DeepSpeed checkpoint size: approximately 4.2 GiB.
- `last.ckpt` is also current internally and approximately 4.2 GiB.
- Retained step checkpoints: 65k, 70k, 75k, 80k, and 85k.

The directory modification time for a DeepSpeed `last.ckpt` directory can look
old even when its internal shard files are current; inspect the files inside
`last.ckpt/checkpoint/` when checking freshness.

### Evaluation exports

Job `3468448` consolidated both final DeepSpeed checkpoints into ordinary
PyTorch state dictionaries on Lorax-local storage:

| Model | Export | Size | Load verification |
|---|---|---:|---|
| Original 85k | `/data/cicicai/flow_slm/evaluation/exports/original-85k/pytorch_model.bin` | 2,261,544,571 bytes | Strict load passed |
| Block-8 85k | `/data/cicicai/flow_slm/evaluation/exports/block8-85k/pytorch_model.bin` | 2,339,145,259 bytes | Strict load passed |

The evaluators and audio-generation jobs each performed their own strict load;
no missing or unexpected learned keys were silently accepted.

## Downstream evaluation results

The reference values below are from the Flow-SLM-270M row recorded in
[`paper_targets.yaml`](paper_targets.yaml).

| Metric | Paper Flow-SLM-270M | Released checkpoint baseline | Our trained original 85k | Difference from paper |
|---|---:|---:|---:|---:|
| sWUGGY test | 68.7 | Not run | **69.99** | **+1.29** |
| sBLIMP test | 57.3 | Not run | **59.63** | **+2.33** |
| SALMon consistency | 70.8 | 66.4 | **72.33** | **+1.53** |
| SALMon sentiment alignment | 60.0 | 58.5 | **54.50** | **-5.50** |
| SALMon background alignment | 55.5 | 54.0 | **54.50** | **-1.00** |
| genPPL | 173.3 | Not run | Not run | N/A |
| Speaker similarity | 0.36 | Not run | Not run | N/A |
| emotion2vec FSD | 3.23 | Not run | Not run | N/A |
| WavLM FSD | 1235.4 | Not run | Not run | N/A |

The sWUGGY/sBLIMP values above use the exact ZeroSpeech 2021 test split:
40,000 lexical pairs and 63,000 syntactic pairs. sBLIMP is the official
pair-weighted score. Scoring formulas were pinned to ZeroSpeech toolkit commit
`199624adfba52901bab564b076fe7d4a63f47ddb`. Semantic token log likelihood was
used, matching the paper's evaluation protocol and excluding CFM likelihood.

As a split-level cross-check, the original model obtained **70.13% sWUGGY** and
**59.52% sBLIMP** on development. The comparable paper ablation values are
68.3% and 57.1%, differences of +1.83 and +2.42 points respectively.

Raw results:

- [`results/original-85k-zerospeech-test.json`](results/original-85k-zerospeech-test.json)
- [`results/original-85k-zerospeech-dev.json`](results/original-85k-zerospeech-dev.json)
- [`results/original-85k-salmon.json`](results/original-85k-salmon.json)

SALMon covered all eight released partitions (200 pairs each), used BF16
autocast, and pinned the dataset to revision
`9aea707934240138d01cfc1b6f9ed7cb608d99d5`. Consistency is the mean of the six
consistency partitions, exactly as recorded by the evaluator.

### Block-8 diagnostic proxy

Block-8 was intentionally trained without semantic token prediction, so it
cannot produce the semantic-token likelihood used for the paper's downstream
metrics. For diagnostic coverage only, we compared paired samples by a
two-sample Monte Carlo flow-matching-loss preference proxy:

| Proxy metric | Block-8 85k |
|---|---:|
| sWUGGY dev proxy | 49.35 |
| sBLIMP dev proxy | 49.82 |
| SALMon consistency proxy | 59.25 |
| SALMon sentiment proxy | 49.00 |
| SALMon background proxy | 51.00 |

These values are near chance and are **not paper-comparable likelihood
metrics**. They diagnose the limitation of the current Block-8 objective rather
than constitute a failed reproduction of a metric that architecture cannot
compute. Raw outputs are in
[`results/block8-85k-zerospeech-dev-flow-proxy.json`](results/block8-85k-zerospeech-dev-flow-proxy.json)
and
[`results/block8-85k-salmon-flow-proxy.json`](results/block8-85k-salmon-flow-proxy.json).

### What can and cannot currently be claimed

Can be claimed:

- full MLS training completed for our original and Block-8 implementations;
- both runs reached 85,000 finite optimizer steps and produced final
  checkpoints;
- both final trained checkpoints export and load strictly;
- our original 85k checkpoint exceeds the paper's sWUGGY, sBLIMP, and SALMon
  consistency values under the pinned public likelihood evaluators;
- six matched continuation examples were generated and validated.

Cannot be claimed:

- exact reproduction of all reported values, because two SALMon targets were
  missed;
- that Block-8 proxy scores are comparable to semantic-token likelihood;
- that complete Tables I-III were reproduced, because the released repository
  lacks complete pinned implementations for genPPL, speaker similarity,
  emotion2vec FSD, and WavLM FSD;
- that three qualitative samples per model substitute for the paper's full
  500-prompt, four-continuation generation evaluation.

## Audio continuations

Job `3468449` generated three matched-prompt continuations from each final
checkpoint. Each example uses the same 3-second LibriSpeech prompt and seed for
the two models, and extends the waveform to 10 seconds total.

| Setting | Value |
|---|---:|
| Seed | 20260820 |
| Sample rate | 24 kHz, mono |
| ODE solver / steps | Euler / 32 |
| CFG scale | 0.3 |
| Semantic / flow temperature | 0.8 / 0.8 |
| Top-p | 0.95 |

Validation job `3468457` confirmed that every WAV contains 240,000 finite
samples with zero clipped-sample fraction. Original-model RMS ranged from
0.0424 to 0.0731 and peak magnitude from 0.543 to 0.625; Block-8 RMS ranged
from 0.0695 to 0.0858 and peak magnitude from 0.543 to 0.914.

- [Original-model continuations](audio_continuations/original)
- [Block-8 continuations](audio_continuations/block8)
- [Machine-readable generation and validation manifest](audio_continuations/manifest.json)

The six WAVs are versioned in GitHub commit
[`79557c4`](https://github.com/Cicicai379/flow-slm/commit/79557c4).

## Smoke tests and validation

| Test | Job | Result |
|---|---:|---|
| Released checkpoint strict load and forward pass | `3453415` | Completed; all learned keys matched, output shapes validated |
| Original three-step real-audio smoke | `3455962` | Completed; flow losses `1.000`, `1.020`, `0.996`; token losses `37.9`, `38.3`, `36.7` |
| Block-8 three-step real-audio smoke | `3455923` | Completed; flow losses `0.995`, `1.000`, `0.990` |
| Block utility unit tests | local unittest | 3/3 passed |
| Python compilation checks | local | Passed |
| Slurm launcher shell syntax | local | Passed |
| Final checkpoint export and strict load | `3468448` | Completed for both models |
| Matched audio generation | `3468449` | Completed; six WAVs |
| Audio validation and copy | `3468457` | Completed; finite, 10 s, no clipping |
| SALMon final evaluation | `3468450` | Completed; all eight partitions |
| ZeroSpeech dev original + Block proxy | `3468497` | Completed (`0:0`) in 00:55:14 |
| ZeroSpeech exact test original | `3468508` | Completed (`0:0`) in 02:14:57 |
| Final result collection | `3468509` | Completed (`0:0`) |

Smoke-test W&B links:

- Original: <https://wandb.ai/cicicai-university-of-california-berkeley/flow-slm-smoke/runs/flowslm-original-smoke-3455962>
- Block-8: <https://wandb.ai/cicicai-university-of-california-berkeley/flow-slm-smoke/runs/flowslm-block8-smoke-3455923>

## Important implementation and infrastructure fixes

- Added online W&B logging with stable run IDs and node-local buffering.
- Restored paper-era 16 kHz input to 24 kHz Mimi model-side resampling.
- Propagated the configured sample rate through Hugging Face and local-data
  loaders.
- Added local real-audio training support for smoke tests.
- Disabled Lightning's optional model summary to avoid a DeepSpeed Stage-3
  summary incompatibility.
- Set CUDA 12.9, Triton, extension, and matplotlib cache locations explicitly.
- Disabled production autograd anomaly detection after smoke validation.
- Constrained launchers to exactly one node. Without `--nodes=1`, Slurm spread
  generic `--gpus=N` requests across Horton and Lorax, which is incompatible
  with node-local MLS storage.
- Staged and verified the complete MLS Arrow cache on Lorax before training.
- Added frame-level partial-block masks, causal block shifting, vectorized block
  loss computation, within-block attention, and isolated Block-8 configs/tests.

## Job history relevant to debugging

| Job(s) | Outcome | Meaning |
|---|---|---|
| `3455922` | Failed | Original smoke exposed Lightning `DeepSpeedSummary` incompatibility; fixed with `enable_model_summary=False` |
| `3455965`, `3455966` | Cancelled | Scheduler spread each request over Horton and Lorax; replaced with exact single-node jobs |
| `3456006` | Completed | Copied and verified the 659 GiB MLS Arrow cache on Lorax |
| `3456017` | Completed | Retargeted pending production work to Lorax |
| `3456434` | Cancelled after 1:08:10 | Short Block-8 full-MLS validation/backfill; released GPUs for original production startup |
| `3456432` | Completed | Final original training run |
| `3456847` | Completed | Final Block-8 training run, resuming only its own Block-8 checkpoint |
| `3468444` | Failed | Initial export attempt exposed missing `CUDA_HOME`; launcher fixed before successful rerun |
| `3468445` | Reported failed after successful staging | Archive checksum/extraction succeeded; a final `find | head` under `pipefail` caused SIGPIPE and was replaced by explicit count checks |
| `3468448` | Completed | Exported both 85k checkpoints and verified strict loads |
| `3468449`, `3468457` | Completed | Generated, validated, and copied six matched audio continuations |
| `3468450` | Completed | Evaluated original SALMon likelihood and Block-8 diagnostic proxy |
| `3468497` | Completed | Evaluated ZeroSpeech development for original and Block-8 proxy |
| `3468508` | Completed | Evaluated exact ZeroSpeech test for the trained original model |
| `3468509` | Completed | Collected all final JSON results into the repository |

## Reproduction commands

From the repository root:

```bash
# Original production training
/accounts/projects/berkeleynlp/anyaji/safe_submit.sh \
  sbatch reproduction/slurm_train_270m.sbatch

# Block-8 production training
/accounts/projects/berkeleynlp/anyaji/safe_submit.sh \
  sbatch blockwise/slurm_train_block8.sbatch

# Released-checkpoint compatibility smoke
/accounts/projects/berkeleynlp/anyaji/safe_submit.sh \
  sbatch reproduction/slurm_checkpoint_smoke.sbatch

# Final trained-checkpoint SALMon evaluation
/accounts/projects/berkeleynlp/anyaji/safe_submit.sh \
  sbatch reproduction/slurm_eval_trained_salmon.sbatch

# ZeroSpeech development: trained original followed by Block-8 proxy
/accounts/projects/berkeleynlp/anyaji/safe_submit.sh \
  sbatch reproduction/slurm_eval_zerospeech.sbatch

# Exact ZeroSpeech test: trained original
/accounts/projects/berkeleynlp/anyaji/safe_submit.sh \
  sbatch reproduction/slurm_eval_zerospeech_test_original.sbatch

# Matched continuation generation
/accounts/projects/berkeleynlp/anyaji/safe_submit.sh \
  sbatch reproduction/slurm_generate_audio_smoke.sbatch
```

Do not use `/scratch` or `/accounts` for datasets, model checkpoints, W&B
buffers, Hugging Face caches, Triton caches, or extension builds. Only the
existing Python environment executable remains under `/scratch`; high-volume
runtime state belongs in the executing node's `/data/cicicai` tree.

## Remaining work for a complete paper-table claim

1. Obtain or independently reconstruct the exact pinned genPPL,
   speaker-similarity, emotion2vec-FSD, and WavLM-FSD pipelines, which are not
   fully implemented in the released repository.
2. Generate four continuations for each of the paper's 500 LibriSpeech test
   prompts and execute those four metric pipelines.
3. Investigate the 5.50-point SALMon sentiment-alignment gap (seed variance,
   training-data/version drift, and optimizer/runtime differences are possible
   contributors) instead of treating the current result as an exact match.
4. If paper-comparable linguistic evaluation is required for Block-8, add and
   train a semantic-token prediction head; the present flow-loss proxy cannot
   replace it.

## Source files

- Original trainer: [`../trainer.py`](../trainer.py)
- Block trainer: [`../trainer_block.py`](../trainer_block.py)
- Original production launcher: [`slurm_train_270m.sbatch`](slurm_train_270m.sbatch)
- Block production launcher: [`../blockwise/slurm_train_block8.sbatch`](../blockwise/slurm_train_block8.sbatch)
- SALMon evaluator: [`eval_salmon.py`](eval_salmon.py)
- Final trained SALMon evaluator: [`eval_trained_salmon.py`](eval_trained_salmon.py)
- ZeroSpeech evaluator: [`eval_zerospeech_likelihood.py`](eval_zerospeech_likelihood.py)
- Audio manifest: [`audio_continuations/manifest.json`](audio_continuations/manifest.json)
- Raw evaluation results: [`results/`](results)
- Paper targets: [`paper_targets.yaml`](paper_targets.yaml)
- Blockwise design: [`../blockwise/PLAN.md`](../blockwise/PLAN.md)
