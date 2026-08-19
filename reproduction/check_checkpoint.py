#!/usr/bin/env python3
"""Strictly validate a released Flow-SLM checkpoint and run a tiny GPU pass."""

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import munch
import torch
import torchaudio
import yaml

# Executing this file directly puts reproduction/, rather than the repository
# root, on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from trainer import LanguageModeling


EXPECTED_CHECKPOINT_BYTES = {
    "270m": 2_261_502_382,
}

# The released checkpoint was produced before commit 457842d moved MLS
# resampling from MimiEncoder (16 -> 24 kHz) into the dataset loader. This is a
# fixed torchaudio FIR buffer, not a learned parameter. Current code receives
# 24 kHz waveforms and therefore must not restore or apply the old resampler.
KNOWN_MIGRATED_BUFFERS = {
    "gslm_pipeline.ssl_model.resample.kernel": (3, 1, 16),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="conf/270m.yaml")
    parser.add_argument("--artifact", choices=EXPECTED_CHECKPOINT_BYTES, default="270m")
    parser.add_argument("--skip-forward", action="store_true")
    return parser.parse_args()


def build_model(checkpoint, config, artifact="270m", paper_sample_rate_compat=False):
    """Build the current model and strictly load all learned checkpoint state."""
    actual_bytes = os.path.getsize(checkpoint)
    expected_bytes = EXPECTED_CHECKPOINT_BYTES[artifact]
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"Checkpoint size mismatch: got {actual_bytes:,}, expected {expected_bytes:,}"
        )

    with open(config) as handle:
        conf = munch.munchify(yaml.safe_load(handle))

    # flash-attn is not installed in the released environment. Eager attention does
    # not change parameter names and permits strict compatibility validation.
    conf.model.flash_attention = False
    model_args = SimpleNamespace(use_k_future_tokens=4, ignore_eos=True)
    print("constructing model", flush=True)
    model = LanguageModeling(model_args, conf)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if paper_sample_rate_compat:
        encoder = model.gslm_pipeline.ssl_model
        encoder.resample = torchaudio.transforms.Resample(orig_freq=16_000, new_freq=24_000)
        print("restored paper-era model-side 16 kHz -> 24 kHz resampling", flush=True)
    else:
        for key, expected_shape in KNOWN_MIGRATED_BUFFERS.items():
            value = state_dict.pop(key, None)
            if value is None or tuple(value.shape) != expected_shape:
                raise RuntimeError(f"Missing or malformed migrated checkpoint buffer: {key}")
            print(f"validated migrated non-learned buffer: {key} {tuple(value.shape)}", flush=True)
    incompatible = model.load_state_dict(state_dict, strict=True)
    print(f"strict checkpoint load: {incompatible}", flush=True)
    print(
        f"parameters: total={sum(p.numel() for p in model.parameters()):,} "
        f"trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
        flush=True,
    )
    return model


def main():
    args = parse_args()
    model = build_model(args.checkpoint, args.config, args.artifact)

    if args.skip_forward:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("The forward smoke test requires a CUDA device")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model = model.cuda().to(torch.bfloat16).eval()
    wav = torch.zeros(1, 24_000, device="cuda", dtype=torch.bfloat16)
    wav_len = torch.ones(1, device="cuda")
    with torch.no_grad():
        outputs = model.gslm_pipeline(wav, wav_len)
    shapes = [tuple(value.shape) if value is not None else None for value in outputs]
    if any(value is not None and not torch.isfinite(value).all() for value in outputs):
        raise RuntimeError("Non-finite value in checkpoint forward pass")
    print(f"forward shapes: {shapes}", flush=True)


if __name__ == "__main__":
    main()
