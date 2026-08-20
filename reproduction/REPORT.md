# Flow-SLM Reproduction and Blockwise Flow-Matching Report

**Report date:** 2026-08-20  
**Repository:** <https://github.com/Cicicai379/flow-slm>  
**Code snapshot:** [`8173f24`](https://github.com/Cicicai379/flow-slm/commit/8173f24)  
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
has therefore been reproduced operationally. The paper's complete downstream
metric table has **not** yet been reproduced: the newly trained original model
still needs checkpoint export and evaluation on SALMon, sWUGGY, sBLIMP, and the
generation metrics.

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

## Paper targets and evaluation status

The reference values below are from the Flow-SLM-270M row recorded in
[`paper_targets.yaml`](paper_targets.yaml).

| Metric | Paper Flow-SLM-270M | Authors' released checkpoint measured here | Our newly trained checkpoint |
|---|---:|---:|---|
| sWUGGY | 68.7 | Not run | Not run |
| sBLIMP | 57.3 | Not run | Not run |
| SALMon consistency | 70.8 | 66.4 | Not run |
| SALMon sentiment alignment | 60.0 | 58.5 | Not run |
| SALMon background alignment | 55.5 | 54.0 | Not run |
| genPPL | 173.3 | Not run | Not run |
| Speaker similarity | 0.36 | Not run | Not run |
| emotion2vec FSD | 3.23 | Not run | Not run |
| WavLM FSD | 1235.4 | Not run | Not run |

For the three released-checkpoint SALMon numbers, the differences from the
paper are `-4.4`, `-1.5`, and `-1.5` percentage points respectively. The
evaluation used semantic token likelihood only, excluded CFM likelihood as
described by the paper, used BF16 autocast, and pinned the SALMon dataset to
revision `9aea707934240138d01cfc1b6f9ed7cb608d99d5`.

### What can and cannot currently be claimed

Can be claimed:

- full MLS training completed for our original and Block-8 implementations;
- both runs reached 85,000 finite optimizer steps and produced final
  checkpoints;
- the authors' released checkpoint loads strictly after restoring the
  paper-era non-learned resampling buffer;
- the SALMon evaluation path runs end-to-end on that released checkpoint.

Cannot yet be claimed:

- that our newly trained model matches the paper's SALMon scores;
- that the complete Tables I-III results were reproduced;
- that generation metrics match until the Whisper, LLaMA, WavLM-TDNN,
  emotion2vec, and WavLM feature pipelines are pinned and executed.

## Smoke tests and validation

| Test | Job | Result |
|---|---:|---|
| Released checkpoint strict load and forward pass | `3453415` | Completed; all learned keys matched, output shapes validated |
| Original three-step real-audio smoke | `3455962` | Completed; flow losses `1.000`, `1.020`, `0.996`; token losses `37.9`, `38.3`, `36.7` |
| Block-8 three-step real-audio smoke | `3455923` | Completed; flow losses `0.995`, `1.000`, `0.990` |
| Block utility unit tests | local unittest | 3/3 passed |
| Python compilation checks | local | Passed |
| Slurm launcher shell syntax | local | Passed |

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

# Released-checkpoint SALMon baseline
/accounts/projects/berkeleynlp/anyaji/safe_submit.sh \
  sbatch reproduction/slurm_salmon_270m.sbatch
```

Do not use `/scratch` or `/accounts` for datasets, model checkpoints, W&B
buffers, Hugging Face caches, Triton caches, or extension builds. Only the
existing Python environment executable remains under `/scratch`; high-volume
runtime state belongs in the executing node's `/data/cicicai` tree.

## Next steps for a strict paper-result claim

1. Consolidate/export the original 85k DeepSpeed checkpoint into the state-dict
   format expected by the evaluation code.
2. Run strict forward compatibility checks on the exported checkpoint.
3. Evaluate our checkpoint on all SALMon partitions and compare against the
   target row above.
4. Obtain/pin ZeroSpeech 2021 and evaluate sWUGGY and sBLIMP.
5. Generate four continuations for each of the paper's 500 LibriSpeech prompts
   using the recorded inference settings.
6. Pin and execute genPPL, speaker similarity, emotion2vec FSD, and WavLM FSD
   scoring pipelines.
7. Record seeds, evaluator commits, per-metric confidence intervals, and exact
   differences from the paper before claiming reproduction.

## Source files

- Original trainer: [`../trainer.py`](../trainer.py)
- Block trainer: [`../trainer_block.py`](../trainer_block.py)
- Original production launcher: [`slurm_train_270m.sbatch`](slurm_train_270m.sbatch)
- Block production launcher: [`../blockwise/slurm_train_block8.sbatch`](../blockwise/slurm_train_block8.sbatch)
- SALMon evaluator: [`eval_salmon.py`](eval_salmon.py)
- Paper targets: [`paper_targets.yaml`](paper_targets.yaml)
- Blockwise design: [`../blockwise/PLAN.md`](../blockwise/PLAN.md)

