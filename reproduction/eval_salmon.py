#!/usr/bin/env python3
"""Evaluate Flow-SLM semantic likelihood with the official SALMon pairs."""

import argparse

import torch
import torchaudio.functional as audio_functional
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from check_checkpoint import build_model
from utils import batch_pad_right


SALMON_REVISION = "9aea707934240138d01cfc1b6f9ed7cb608d99d5"
PARTS = (
    "bg_alignment",
    "bg_all_consistency",
    "bg_domain_consistency",
    "gender_consistency",
    "rir_consistency",
    "sentiment_alignment",
    "sentiment_consistency",
    "speaker_consistency",
)


class SalmonPairs(Dataset):
    def __init__(self, part):
        self.dataset = load_dataset(
            "slprl/SALMon", part, split="train", revision=SALMON_REVISION
        )

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _audio(row):
        audio = row
        # datasets<=3 yields a dict; newer datasets may expose an AudioDecoder.
        if not isinstance(audio, dict):
            audio = audio.get_all_samples()
            waveform = torch.as_tensor(audio.data, dtype=torch.float32).squeeze(0)
            sample_rate = audio.sample_rate
        else:
            waveform = torch.as_tensor(audio["array"], dtype=torch.float32).squeeze(0)
            sample_rate = audio["sampling_rate"]
        if sample_rate != 16_000:
            waveform = audio_functional.resample(waveform, sample_rate, 16_000)
        return waveform

    def __getitem__(self, index):
        row = self.dataset[index]
        return self._audio(row["positive_audio"]), self._audio(row["negative_audio"])


def collate_pairs(rows):
    positive, negative = zip(*rows)
    return list(positive), list(negative)


def semantic_log_likelihood(model, waveforms, device):
    padded, lengths = batch_pad_right(waveforms)
    padded = padded.to(device=device, dtype=torch.float32)
    lengths = lengths.to(device=device, dtype=torch.float32)
    ids = [str(index) for index in range(len(waveforms))]
    # This follows trainer.py's paper evaluation path. The CFM likelihood is
    # intentionally excluded, as stated in Sec. IV-B of the paper.
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, _, token_loss, _ = model.forward((ids, padded, lengths), reduction="utterance")
    return -token_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="conf/270m.yaml")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--parts", nargs="+", choices=PARTS, default=list(PARTS))
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SALMon evaluation requires a CUDA device")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.device("cuda")
    # Match Lightning's `bf16-mixed`: keep non-OpenELM parameters in FP32 and
    # autocast the forward pass, rather than converting the whole model to BF16.
    model = build_model(
        args.checkpoint, args.config, paper_sample_rate_compat=True
    ).to(device).eval()

    scores = {}
    for part in args.parts:
        loader = DataLoader(
            SalmonPairs(part),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_pairs,
        )
        correct = []
        with torch.no_grad():
            for positive, negative in loader:
                pos_score = semantic_log_likelihood(model, positive, device)
                neg_score = semantic_log_likelihood(model, negative, device)
                batch_score = torch.where(
                    pos_score > neg_score,
                    torch.ones_like(pos_score),
                    torch.where(
                        pos_score == neg_score,
                        torch.full_like(pos_score, 0.5),
                        torch.zeros_like(pos_score),
                    ),
                )
                correct.append(batch_score.cpu())
        scores[part] = torch.cat(correct).float().mean().item() * 100
        print(f"SALMon {part}: {scores[part]:.1f}", flush=True)

    consistency_parts = [part for part in PARTS if "consistency" in part]
    consistency = sum(scores[part] for part in consistency_parts) / len(consistency_parts)
    print(f"SALMon consistency: {consistency:.1f}", flush=True)
    print(f"SALMon sentiment alignment: {scores['sentiment_alignment']:.1f}", flush=True)
    print(f"SALMon background alignment: {scores['bg_alignment']:.1f}", flush=True)


if __name__ == "__main__":
    main()
