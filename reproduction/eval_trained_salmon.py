#!/usr/bin/env python3
"""Evaluate the final self-trained checkpoints on SALMon paired audio.

The original model is scored with semantic-token log likelihood, matching the
paper.  Block-8 has no token prediction head; for that model this script emits
an explicitly non-comparable Monte Carlo flow-loss preference diagnostic.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import munch
import torch
import torchaudio.functional as audio_functional
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from trainer import LanguageModeling
from trainer_block import BlockLanguageModeling
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
    def _audio(audio):
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


def load_model(kind, checkpoint, config, device):
    with open(config) as handle:
        conf = munch.munchify(yaml.safe_load(handle))
    args = SimpleNamespace(
        use_k_future_tokens=4 if kind == "original" else 0,
        ignore_eos=True,
    )
    cls = LanguageModeling if kind == "original" else BlockLanguageModeling
    print(f"constructing {kind} model", flush=True)
    model = cls(args, conf)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    incompatible = model.load_state_dict(state, strict=True)
    print(f"strict checkpoint load: {incompatible}", flush=True)
    return model.to(device).eval()


def padded_batch(waveforms, device):
    padded, lengths = batch_pad_right(waveforms)
    ids = [str(index) for index in range(len(waveforms))]
    return ids, padded.to(device=device, dtype=torch.float32), lengths.to(device).float()


def original_score(model, waveforms, device):
    _, padded, lengths = padded_batch(waveforms, device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, _, _, token_logits, tokens, token_mask = model._run_pipeline(
            padded, lengths, eval_mode=True
        )
        token_loss = model._compute_token_loss(
            token_logits, tokens, token_mask, training=False
        )
        length = token_mask.shape[1]
        token_mask = token_mask * (
            tokens[:, :length].squeeze(2) != model.gslm_pipeline.eos_token_index
        )
        utterance_loss = (token_loss * token_mask).sum(1) / token_mask.sum(1)
    return -utterance_loss


def block_score(model, waveforms, device, mc_samples, seed):
    batch = padded_batch(waveforms, device)
    scores = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for repeat in range(mc_samples):
            torch.manual_seed(seed + repeat)
            torch.cuda.manual_seed_all(seed + repeat)
            _, flow_loss, _, _ = model.forward(batch, reduction="utterance")
            scores.append(-flow_loss)
    return torch.stack(scores).mean(0)


def pair_accuracy(pos_score, neg_score):
    return torch.where(
        pos_score > neg_score,
        torch.ones_like(pos_score),
        torch.where(
            pos_score == neg_score,
            torch.full_like(pos_score, 0.5),
            torch.zeros_like(pos_score),
        ),
    )


def write_results(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("original", "block8"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mc-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--parts", nargs="+", choices=PARTS, default=list(PARTS))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("SALMon evaluation requires CUDA")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    model = load_model(args.kind, args.checkpoint, args.config, device)

    result = {
        "model": args.kind,
        "checkpoint": args.checkpoint,
        "salmon_revision": SALMON_REVISION,
        "seed": args.seed,
        "score_type": (
            "semantic_token_log_likelihood"
            if args.kind == "original"
            else "mc_flow_matching_loss_preference_proxy_not_paper_comparable"
        ),
        "mc_samples": 1 if args.kind == "original" else args.mc_samples,
        "parts": {},
    }
    for part_index, part in enumerate(args.parts):
        loader = DataLoader(
            SalmonPairs(part),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_pairs,
        )
        correct = []
        for batch_index, (positive, negative) in enumerate(loader):
            if args.kind == "original":
                pos_score = original_score(model, positive, device)
                neg_score = original_score(model, negative, device)
            else:
                pair_seed = args.seed + part_index * 1_000_000 + batch_index * 100
                # Reset to the same sequence for the two sides to reduce the
                # variance introduced by stochastic flow-matching scoring.
                pos_score = block_score(model, positive, device, args.mc_samples, pair_seed)
                neg_score = block_score(model, negative, device, args.mc_samples, pair_seed)
            correct.append(pair_accuracy(pos_score, neg_score).cpu())
        values = torch.cat(correct).float()
        result["parts"][part] = {
            "accuracy_percent": values.mean().item() * 100,
            "pairs": values.numel(),
        }
        write_results(args.output, result)
        print(
            f"SALMon {part}: {result['parts'][part]['accuracy_percent']:.3f} "
            f"({values.numel()} pairs)",
            flush=True,
        )

    scores = {key: value["accuracy_percent"] for key, value in result["parts"].items()}
    consistency_parts = [part for part in PARTS if "consistency" in part]
    if all(part in scores for part in consistency_parts):
        result["summary"] = {
            "consistency_percent": sum(scores[part] for part in consistency_parts)
            / len(consistency_parts),
            "sentiment_alignment_percent": scores.get("sentiment_alignment"),
            "background_alignment_percent": scores.get("bg_alignment"),
        }
    write_results(args.output, result)
    print(json.dumps(result.get("summary", {}), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
