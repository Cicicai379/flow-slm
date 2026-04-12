import argparse
from pathlib import Path
import tqdm
import torch
import torch.nn.functional as F
import torchaudio
import yaml
import munch

from trainer_block import BlockLanguageModeling
from inference import Processor, load_audio_list, save_wav, load_conf, WhisperWrapper


class BlockSampler(torch.nn.Module):
    """Autoregressive block-level sampler using BlockFlowLoss."""

    def __init__(self, gslm_pipeline, block_flow_loss, frame_rate=12.5, silence_indices=None):
        super().__init__()
        self.gslm_pipeline = gslm_pipeline
        self.block_flow_loss = block_flow_loss
        self.conf = gslm_pipeline.conf
        self.ssl_dim = self.conf.model.ssl_dim
        self.reduction_factor = self.conf.model.reduction_factor
        self.block_size = gslm_pipeline.block_size
        self.feature_dim = self.ssl_dim * self.reduction_factor
        self.block_dim = self.block_size * self.feature_dim
        self.frame_rate = frame_rate
        self.silence_indices = silence_indices or [1049, 127, 1880, 1492, 972, 1031, 395, 2029, 581, 175, 1926, 407, 1316]

    def _sample_token(self, logits, topp=None, temperature=1.0, penalize_silence=False, penalize_weight=10.0):
        if temperature != 1.0:
            logits = logits / temperature
        if penalize_silence and self.silence_indices:
            logits = logits.clone()
            logits[:, self.silence_indices] -= penalize_weight
        if topp is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > topp
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            sorted_logits[sorted_indices_to_remove] = float("-inf")
            logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)
        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)  # [B, 1]

    def sample(self, batch_size=1, max_len=15, ode_steps=64, device="cuda",
               token_temperature=1.0, temperature=1.0, prompts=None,
               solver="euler", eos_aux_token=None, cfg_scale=0.3,
               topp=0.95, penalize_silence=False, penalize_weight=10.0):
        """Generate blocks autoregressively.

        Args:
            prompts: [B, T_prompt, feature_dim] normalized, reduced prompt features.
        Returns:
            frames [B, T_total, feature_dim] (prompt + generated), stop_steps [B] in reduced frames.
        """
        max_infer_frames = round(max_len * self.frame_rate / self.reduction_factor)
        max_infer_blocks = (max_infer_frames + self.block_size - 1) // self.block_size

        if prompts is not None:
            B = prompts.shape[0]
            T_prompt = prompts.shape[1]
            num_prompt_blocks = (T_prompt + self.block_size - 1) // self.block_size
            padded_T = num_prompt_blocks * self.block_size
            padded = prompts.to(device).to(torch.bfloat16)
            if padded_T > T_prompt:
                pad = torch.zeros(B, padded_T - T_prompt, self.feature_dim, device=device, dtype=torch.bfloat16)
                padded = torch.cat([padded, pad], dim=1)
            block_sequence = padded.reshape(B, num_prompt_blocks, self.block_dim)
        else:
            B = batch_size
            num_prompt_blocks = 1
            null = self.gslm_pipeline.null_block.detach().unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
            block_sequence = null.to(device).to(torch.bfloat16)

        has_ended = torch.zeros(B, dtype=torch.bool, device=device)
        stop_steps = torch.zeros(B, dtype=torch.int32, device=device)
        generated_blocks = []

        for block_idx in range(max_infer_blocks):
            attn_mask = torch.ones(B, block_sequence.shape[1], device=device, dtype=torch.bool)
            block_reprs, aux_output = self.gslm_pipeline.decoder(block_sequence, attention_mask=attn_mask)

            z = block_reprs[:, -1]  # [B, decoder_dim]

            # EOS detection via aux token prediction
            if eos_aux_token is not None and aux_output is not None:
                token_logits = aux_output[:, -1]  # [B, aux_output_dim]
                tokens = self._sample_token(token_logits, topp=topp, temperature=token_temperature,
                                            penalize_silence=penalize_silence, penalize_weight=penalize_weight)
                is_eos = tokens.squeeze(1) == eos_aux_token
                end_at_this_step = is_eos & (stop_steps == 0)
                stop_steps[end_at_this_step] = (num_prompt_blocks + block_idx) * self.block_size
                has_ended |= end_at_this_step

            if has_ended.all():
                break

            new_block = self.block_flow_loss.sample(z, steps=ode_steps, temperature=temperature,
                                                    solver=solver, cfg_scale=cfg_scale)
            generated_blocks.append(new_block.to(torch.bfloat16))
            block_sequence = torch.cat([block_sequence, new_block.unsqueeze(1).to(torch.bfloat16)], dim=1)

        if not has_ended.all():
            stop_steps[stop_steps == 0] = (num_prompt_blocks + len(generated_blocks)) * self.block_size

        if not generated_blocks:
            out = prompts.to(device).to(torch.bfloat16) if prompts is not None else \
                  torch.zeros(B, 0, self.feature_dim, device=device, dtype=torch.bfloat16)
            return out, stop_steps

        all_gen = torch.stack(generated_blocks, dim=1)  # [B, n_gen, block_dim]
        gen_frames = all_gen.reshape(B, len(generated_blocks) * self.block_size, self.feature_dim)

        if prompts is not None:
            out_frames = torch.cat([prompts.to(device).to(torch.bfloat16), gen_frames], dim=1)
        else:
            out_frames = gen_frames

        return out_frames, stop_steps


def load_model(args, conf, device="cuda"):
    model_args = type("Args", (), {})()
    lm = BlockLanguageModeling(model_args, conf)
    state_dict = torch.load(args.ckpt_path, map_location="cpu")
    if "epoch" in state_dict:
        state_dict = state_dict["state_dict"]
    lm.load_state_dict(state_dict, strict=False)
    lm = lm.to(device).to(torch.bfloat16)
    return lm


def prepare_sampler_and_processor(lm, conf, args, device="cuda"):
    frame_rate = 12.5
    sampler = BlockSampler(lm.gslm_pipeline, lm.loss_fn, frame_rate=frame_rate).to(device)
    processor = Processor(conf, device=device)
    processor.load_vocoder_mimi()
    processor.load_ssl_model(lm.gslm_pipeline.ssl_model.to(torch.bfloat16))
    processor.load_statistics(lm.gslm_pipeline.mean, lm.gslm_pipeline.std)
    return sampler, processor


def run_conditional(args, sampler, processor, prompt_wavs):
    codec_size = 2048
    for prompt_idx, (prompt_id, wav, duration) in enumerate(prompt_wavs):
        reduced_feats = processor.get_ssl_feats(wav, duration, duplicate=args.batch_size)
        reduced_feats = reduced_feats.to(torch.bfloat16)

        for batch_idx in range(args.samples_per_prompt // args.batch_size):
            with torch.no_grad():
                eos_aux_token = codec_size + processor.conf.model.n_special_tokens - 1 \
                    if getattr(processor.conf.optimizer, "token_loss_weight", 0) > 0 else None
                samples, stop_steps = sampler.sample(
                    batch_size=args.batch_size, max_len=args.max_len, ode_steps=args.ode_steps,
                    token_temperature=args.token_temperature, temperature=args.temperature,
                    prompts=reduced_feats, solver=args.solver, eos_aux_token=eos_aux_token,
                    cfg_scale=args.cfg_scale, topp=args.topp,
                    penalize_silence=args.penalize_silence, penalize_weight=args.penalize_weight,
                )

            samples = processor.unmerge_and_unnormalize(samples)
            wavs = processor.batch_vocoding(samples, stop_steps, args.num_quantizers)

            for i, wav in enumerate(wavs):
                yield prompt_id, wav, processor.sample_rate


def run_unconditional(args, sampler, processor):
    codec_size = 2048
    samples_to_generate = args.n_samples
    batch_size = args.batch_size
    generated = 0
    while generated < samples_to_generate:
        cur_bs = min(batch_size, samples_to_generate - generated)
        with torch.no_grad():
            eos_aux_token = codec_size + processor.conf.model.n_special_tokens - 1 \
                if getattr(processor.conf.optimizer, "token_loss_weight", 0) > 0 else None
            samples, stop_steps = sampler.sample(
                batch_size=cur_bs, max_len=args.max_len, ode_steps=args.ode_steps,
                token_temperature=args.token_temperature, temperature=args.temperature,
                solver=args.solver, eos_aux_token=eos_aux_token, cfg_scale=args.cfg_scale,
                topp=args.topp, penalize_silence=args.penalize_silence, penalize_weight=args.penalize_weight,
            )

        samples = processor.unmerge_and_unnormalize(samples)
        wavs = processor.batch_vocoding(samples, stop_steps, args.num_quantizers)
        for wav in wavs:
            yield str(generated), wav, processor.sample_rate
        generated += cur_bs


def parse_args():
    parser = argparse.ArgumentParser(description="Block diffusion GSLM inference")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Block model checkpoint")
    parser.add_argument("--conf_path", type=str, required=True, help="Config file")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--samples_per_prompt", type=int, default=16)
    parser.add_argument("--prompt_dir", type=str, default=None)
    parser.add_argument("--prompt_csv", type=str, default=None)
    parser.add_argument("--max_len", type=float, default=30)
    parser.add_argument("--ode_steps", type=int, default=32)
    parser.add_argument("--topp", type=float, default=0.95)
    parser.add_argument("--penalize_silence", action="store_true")
    parser.add_argument("--penalize_weight", type=float, default=10.0)
    parser.add_argument("--token_temperature", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--cfg_scale", type=float, default=0.3)
    parser.add_argument("--solver", type=str, default="euler")
    parser.add_argument("--save_wav", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--asr", action="store_true")
    parser.add_argument("--download_whisper_root", type=str, default=None)
    parser.add_argument("--save_transcription", action="store_true")
    parser.add_argument("--num_quantizers", type=int, default=16)
    parser.add_argument("--sr", type=int, default=24000)
    return parser.parse_args()


def main():
    args = parse_args()
    conf = load_conf(args.conf_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"loading {args.ckpt_path}")
    lm = load_model(args, conf, device=device)
    lm.eval()

    sampler, processor = prepare_sampler_and_processor(lm, conf, args, device=device)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    transcription_file = None
    if args.save_transcription and args.output_dir:
        transcription_file = open(Path(args.output_dir) / "transcriptions.csv", "w")

    asr_model = None
    if args.asr:
        try:
            asr_model = WhisperWrapper(model_card="large-v3-turbo", device=device,
                                       download_root=args.download_whisper_root)
        except Exception as e:
            print(f"Whisper unavailable: {e}")

    prompt_wavs = None
    if args.prompt_csv is not None and args.prompt_dir is not None:
        prompt_wavs = load_audio_list(args.prompt_dir, args.prompt_csv, target_sample_rate=args.sr)

    if prompt_wavs is None:
        gen_iter = run_unconditional(args, sampler, processor)
    else:
        gen_iter = run_conditional(args, sampler, processor, prompt_wavs)

    for idx, (prompt_id, wav, sr) in enumerate(tqdm.tqdm(gen_iter)):
        if args.save_wav and args.output_dir:
            if prompt_wavs is not None:
                sample_idx = idx % args.samples_per_prompt
                out_path = Path(args.output_dir) / f"{prompt_id}_{sample_idx:04d}.wav"
            else:
                out_path = Path(args.output_dir) / f"{idx:04d}.wav"
            save_wav(wav, str(out_path), sr)

        if asr_model is not None:
            try:
                res = asr_model.transcribe(wav)
            except Exception as e:
                print("ASR failed:", e)
                res = None
        else:
            res = None

        if transcription_file is not None and res is not None:
            print(f"{prompt_id}\t{res.text}", file=transcription_file, flush=True)

    if transcription_file is not None:
        transcription_file.close()


if __name__ == "__main__":
    main()
