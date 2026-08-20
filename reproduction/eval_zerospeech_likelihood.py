#!/usr/bin/env python3
"""Score ZeroSpeech-2021 sWUGGY/sBLIMP audio and aggregate dev metrics."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import munch
import pandas as pd
import torch
import torchaudio
import torchaudio.functional as audio_functional
import yaml
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from trainer import LanguageModeling
from trainer_block import BlockLanguageModeling
from utils import batch_pad_right

OFFICIAL_SCORER_COMMIT = "199624adfba52901bab564b076fe7d4a63f47ddb"


class AudioFiles(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        waveform, sample_rate = torchaudio.load(str(path))
        waveform = waveform.mean(0)
        if sample_rate != 16_000:
            waveform = audio_functional.resample(waveform, sample_rate, 16_000)
        return path.stem, waveform


def collate_audio(rows):
    names, waveforms = zip(*rows)
    return list(names), list(waveforms)


def find_dataset_root(candidate):
    candidate = candidate.resolve()
    if (candidate / "lexical/dev/gold.csv").is_file():
        return candidate
    matches = list(candidate.glob("**/lexical/dev/gold.csv"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one ZeroSpeech dataset below {candidate}, found {matches}")
    return matches[0].parents[2]


def load_model(kind, checkpoint, config, device):
    with open(config) as handle:
        conf = munch.munchify(yaml.safe_load(handle))
    model_args = SimpleNamespace(
        use_k_future_tokens=4 if kind == "original" else 0,
        ignore_eos=True,
    )
    cls = LanguageModeling if kind == "original" else BlockLanguageModeling
    model = cls(model_args, conf)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    incompatible = model.load_state_dict(state, strict=True)
    print(f"strict checkpoint load: {incompatible}", flush=True)
    return model.to(device).eval()


def make_batch(names, waveforms, device):
    padded, lengths = batch_pad_right(waveforms)
    return names, padded.to(device, dtype=torch.float32), lengths.to(device).float()


def score_original(model, names, waveforms, device):
    _, padded, lengths = make_batch(names, waveforms, device)
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


def score_block(model, names, waveforms, device, repeats, seed):
    batch = make_batch(names, waveforms, device)
    values = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for repeat in range(repeats):
            torch.manual_seed(seed + repeat)
            torch.cuda.manual_seed_all(seed + repeat)
            _, flow_loss, _, _ = model.forward(batch, reduction="utterance")
            values.append(-flow_loss)
    return torch.stack(values).mean(0)


def read_scores(path):
    scores = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            name, value = line.rsplit(maxsplit=1)
            scores[name] = float(value)
    return scores


def score_directory(model, kind, audio_dir, output, device, batch_size, repeats, seed):
    paths = sorted(audio_dir.glob("*.wav"))
    if not paths:
        raise RuntimeError(f"No wav files found in {audio_dir}")
    existing = read_scores(output)
    remaining = [path for path in paths if path.stem not in existing]
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"{audio_dir}: {len(paths)} total, {len(remaining)} remaining", flush=True)
    loader = DataLoader(
        AudioFiles(remaining), batch_size=batch_size, shuffle=False,
        num_workers=8, collate_fn=collate_audio,
    )
    with output.open("a") as handle:
        for batch_index, (names, waveforms) in enumerate(loader):
            if kind == "original":
                scores = score_original(model, names, waveforms, device)
            else:
                scores = score_block(
                    model, names, waveforms, device, repeats,
                    seed + batch_index * 100,
                )
            for name, value in zip(names, scores.float().cpu().tolist()):
                handle.write(f"{name} {value:.10g}\n")
            handle.flush()
            if batch_index % 100 == 0:
                print(f"{audio_dir.name}: scored {batch_index * batch_size + len(names)}", flush=True)
    final = read_scores(output)
    expected = {path.stem for path in paths}
    if set(final) != expected:
        raise RuntimeError(
            f"Score manifest mismatch for {audio_dir}: "
            f"missing={len(expected - set(final))}, extra={len(set(final) - expected)}"
        )


def load_gold_and_scores(gold_path, score_path):
    gold = pd.read_csv(gold_path, header=0, index_col="filename")
    score = pd.read_csv(
        score_path, sep=" ", header=None, names=["filename", "score"], index_col="filename"
    )
    if set(gold.index) != set(score.index):
        raise RuntimeError("Gold and submitted filenames differ")
    return gold.join(score).reset_index()


def aggregate_lexical(gold_path, score_path):
    data = load_gold_and_scores(gold_path, score_path)
    if data.loc[data.correct == 0, "word"].isnull().all():
        data["word"] = data["word"].fillna(data["phones"])
    keys = ["voice", "id"]
    words = data[data.correct == 1].set_index(keys)
    nonwords = data[data.correct == 0].set_index(keys)
    pairs = words.join(nonwords, lsuffix="_word", rsuffix="_nonword", how="inner")
    pairs["correct_score"] = (
        (pairs.score_word > pairs.score_nonword).astype(float)
        + 0.5 * (pairs.score_word == pairs.score_nonword).astype(float)
    )
    by_id = pairs.groupby("id").agg(
        score=("correct_score", "mean"), frequency=("frequency_word", "first")
    )
    bands = pd.cut(
        by_id.frequency, [0, 1, 5, 20, 100, float("inf")],
        labels=["oov", "1-5", "6-20", "21-100", ">100"], right=False,
    )
    by_frequency = by_id.score.groupby(bands, observed=False).agg(["count", "mean"])
    invocab = by_id[by_id.frequency > 0].score.mean()
    return {
        "all_percent": float(by_id.score.mean() * 100),
        "in_vocab_percent": float(invocab * 100),
        "pairs": len(by_id),
        "by_frequency_percent": {
            str(index): float(row["mean"] * 100) for index, row in by_frequency.iterrows()
        },
    }


def aggregate_syntactic(gold_path, score_path):
    data = load_gold_and_scores(gold_path, score_path)
    keys = ["voice", "type", "subtype", "id"]
    grammatical = data[data.correct == 1].set_index(keys)
    ungrammatical = data[data.correct == 0].set_index(keys)
    pairs = grammatical.join(ungrammatical, lsuffix="_good", rsuffix="_bad", how="inner")
    pairs["correct_score"] = (
        (pairs.score_good > pairs.score_bad).astype(float)
        + 0.5 * (pairs.score_good == pairs.score_bad).astype(float)
    )
    by_item = pairs.groupby(["type", "subtype", "id"]).correct_score.mean().reset_index()
    by_type = by_item.groupby("type").correct_score.mean()
    return {
        # This is the general score exposed by the pinned official toolkit.
        "official_pair_weighted_percent": float(by_item.correct_score.mean() * 100),
        # Also retain the broad-category macro average described on the task page.
        "broad_type_macro_percent": float(by_type.mean() * 100),
        "pairs": len(by_item),
        "by_type_percent": {str(key): float(value * 100) for key, value in by_type.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("original", "block8"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mc-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("ZeroSpeech likelihood evaluation requires CUDA")

    root = find_dataset_root(args.dataset_root)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = load_model(args.kind, args.checkpoint, args.config, device)
    submission = args.output_root / args.kind / "submission"

    for task in ("lexical", "syntactic"):
        score_directory(
            model, args.kind, root / task / args.split,
            submission / task / f"{args.split}.txt",
            device, args.batch_size, args.mc_samples, args.seed,
        )

    result = {
        "model": args.kind,
        "split": args.split,
        "checkpoint": args.checkpoint,
        "score_type": (
            "semantic_token_log_likelihood"
            if args.kind == "original"
            else "mc_flow_matching_loss_proxy_not_paper_comparable"
        ),
        "mc_samples": 1 if args.kind == "original" else args.mc_samples,
        "official_zerospeech2021_scorer_commit": OFFICIAL_SCORER_COMMIT,
        "sWUGGY": aggregate_lexical(
            root / f"lexical/{args.split}/gold.csv",
            submission / f"lexical/{args.split}.txt",
        ),
        "sBLIMP": aggregate_syntactic(
            root / f"syntactic/{args.split}/gold.csv",
            submission / f"syntactic/{args.split}.txt",
        ),
    }
    result_path = args.output_root / args.kind / f"zerospeech-{args.split}-results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
