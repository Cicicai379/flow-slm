# Paper reproduction

This directory isolates the original Flow-SLM reproduction from the later
block-decoding and SPiDR experiments in the repository.

`paper_targets.yaml` records the training/inference protocol and every Flow-SLM
number reported in Tables I--III of arXiv:2508.09350. These are reference
targets, not assumed pass results.

The first gate is strict compatibility with the authors' released checkpoint:

```bash
/accounts/projects/berkeleynlp/anyaji/safe_submit.sh \
  sbatch --gpus=H200:1 reproduction/slurm_checkpoint_smoke.sbatch
```

Run the official SALMon pairs with the semantic likelihood used in the paper:

```bash
/accounts/projects/berkeleynlp/anyaji/safe_submit.sh \
  sbatch --gpus=H200:1 reproduction/slurm_salmon_270m.sbatch
```

Both jobs are pinned to `lorax` and keep checkpoints and caches under
`/data/cicicai/flow_slm/paper_reproduction`. The shared project filesystem is
used only for code, logs, and final results.

The released 270M checkpoint predates commit `457842d` and contains the fixed
torchaudio buffer `gslm_pipeline.ssl_model.resample.kernel`. That commit moved
16-to-24 kHz resampling from `MimiEncoder` into the dataset loader. The generic
checker validates and removes this one non-learned migrated buffer. Paper
evaluation instead restores the old model-side operation and loads the saved
kernel, requiring a strict match for every checkpoint key.

The original base training run requires the complete 705 GB MLS-English corpus,
one epoch (about 85k steps), and global batch size 128. In this repository that
maps to `conf/270m.yaml`, four workers/GPUs, and batch size 32 per process:

```bash
python trainer.py \
  --conf conf/270m.yaml \
  --save_path /path/to/checkpoints \
  --hf_training_data \
  --training_data MLSEn \
  --strategy deepspeed_stage_3
```

The cluster launcher uses four H200s (batch 32 per process = global batch 128),
the paper-era 16 kHz input plus checkpointed model-side 24 kHz resampler, safe
W&B resumption, and node-local thidwick storage. Stage MLS first with a CPU-only
job, then make training depend on successful staging:

```bash
prep_job=$(/accounts/projects/berkeleynlp/anyaji/safe_submit.sh \
  sbatch --parsable reproduction/slurm_prepare_mls.sbatch | tail -n1)
/accounts/projects/berkeleynlp/anyaji/safe_submit.sh \
  sbatch --gpus=4 --dependency="afterok:${prep_job}" \
  reproduction/slurm_train_270m.sbatch
```

It deliberately exits before model construction unless `wandb.login(verify=True)`
succeeds. The W&B URL is printed to the Slurm log at startup.

Do not use `MLSEn+people` for the base paper model; that selects the extended
65k-hour data mixture. A complete reproduction also requires the official
ZeroSpeech 2021 and SALMon test sets. The paper's random 500-prompt test mixture
is present as `prompts/test_libri.csv` (244 test-clean and 256 test-other). The
released repository does not include the final generation-metric scoring
implementations, so those metrics cannot be claimed as reproduced until the
Whisper, LLaMA, WavLM-TDNN, emotion2vec, and WavLM feature pipelines are pinned.
