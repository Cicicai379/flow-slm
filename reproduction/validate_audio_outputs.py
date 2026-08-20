#!/usr/bin/env python3
"""Validate generated WAVs, copy a small report set, and write a manifest."""

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
import torchaudio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    records = []
    for source in sorted(args.input_root.glob("*/*.wav")):
        model = source.parent.name
        waveform, sample_rate = torchaudio.load(str(source))
        finite = bool(torch.isfinite(waveform).all())
        peak = waveform.abs().max().item()
        rms = waveform.square().mean().sqrt().item()
        duration = waveform.shape[-1] / sample_rate
        clipped_fraction = (waveform.abs() >= 0.999).float().mean().item()
        if not finite or not math.isfinite(peak) or duration <= 0:
            raise RuntimeError(f"Invalid generated audio: {source}")

        destination_dir = args.output_root / model
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / source.name
        shutil.copy2(source, destination)
        records.append(
            {
                "model": model,
                "file": str(destination.relative_to(args.output_root)),
                "sample_rate_hz": sample_rate,
                "channels": waveform.shape[0],
                "samples": waveform.shape[-1],
                "duration_seconds": duration,
                "peak_absolute": peak,
                "rms": rms,
                "clipped_fraction": clipped_fraction,
                "all_finite": finite,
            }
        )

    if len(records) != 6:
        raise RuntimeError(f"Expected six smoke continuations, found {len(records)}")
    manifest = {
        "source": str(args.input_root),
        "generation": {
            "prompt_seconds": 3.0,
            "total_seconds": 10.0,
            "ode_steps": 32,
            "top_p": 0.95,
            "token_temperature": 0.8,
            "flow_temperature": 0.8,
            "cfg_scale": 0.3,
            "solver": "euler",
            "seed": 20260820,
        },
        "files": records,
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
