import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
try:
    from utils import length_to_mask
except ImportError:
    def length_to_mask(lengths, max_len=None, dtype=torch.bool):
        if max_len is None:
            max_len = lengths.max()
        mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
        return mask.to(dtype)

try:
    from model import MimiEncoder
except ImportError:
    print("Warning: MimiEncoder not available, will use dummy encoder")
    MimiEncoder = None

from model_block import ELMBlockDecoderWrapper

try:
    from model_utils import reduce_features, split_features
except ImportError:
    def reduce_features(features, factor, pad=False):
        B, T, F = features.shape
        if pad and T % factor != 0:
            pad_len = factor - (T % factor)
            features = torch.cat([features, features[:, :pad_len]], dim=1)
            T += pad_len
        return features[:, ::factor]

    def split_features(features, factor):
        B, T, F = features.shape
        return features.view(B, T * factor, F // factor)
from transformers import AutoModelForCausalLM


class GSLMBlockPipeline(nn.Module):
    """Block diffusion pipeline for joint block modeling."""
    def __init__(self, conf, args):
        super().__init__()
        self.conf = conf
        self.args = args

        # Block size for joint modeling
        self.block_size = getattr(self.conf.model, "block_size", 8)

        if hasattr(self.conf.model, "ssl_model") and self.conf.model.ssl_model == "mimi":
            n_quantizers = getattr(self.conf.model, "n_quantizers", 0)
            try:
                if MimiEncoder is not None:
                    self.ssl_model = MimiEncoder(freeze=self.conf.model.freeze, n_quantizers=n_quantizers)
                    print("✓ MimiEncoder initialized")
                else:
                    raise ImportError("MimiEncoder not available")
            except Exception as e:
                print(f"Warning: Could not initialize MimiEncoder ({e}), using dummy")
                self.ssl_model = None

        # Initialize decoder model
        if "OpenELM" in self.conf.model.decoder:
            model_name = f"apple/{self.conf.model.decoder}"
        else:
            raise NotImplementedError(f"Decoder model {self.conf.model.decoder} not supported.")

        attn_implementation = "flash_attention_2" if self.conf.model.flash_attention else "eager"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() or attn_implementation == 'flash_attention_2' else torch.float32
        decoder_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True
        )

        # Initialize normalization (moved to helper)
        self._init_normalization()
        # Initialize remaining model components (moved to helper)
        self._init_model_components(decoder_model)

        # Initialize embeddings (moved to helper)
        self._init_embeddings()

    @property
    def _decoder_model(self):
        return self.decoder.lm

    def _init_normalization(self):
        """Load and register static normalization buffers if configured.

        This was previously in __init__; extracted for clarity and reuse.
        """
        if hasattr(self.conf.model, "norm") and self.conf.model.norm == "static":
            mean = np.load(self.conf.model.mean_path)
            self.register_buffer('mean', torch.from_numpy(mean).float())
            std = np.load(self.conf.model.std_path)
            self.register_buffer('std', torch.from_numpy(std).float())
    
    def _init_model_components(self, decoder_model):
        """Initialize dims, aux outputs, decoder wrapper and related model components."""
        ssl_dim, reduction_factor = self.conf.model.ssl_dim, self.conf.model.reduction_factor
        if self.conf.optimizer.loss_function == "BLOCK_FM":
            # For block modeling: each block is flattened to [block_size * ssl_dim * reduction_factor]
            self.feature_dim = ssl_dim * reduction_factor
            self.block_dim = self.block_size * self.feature_dim  # Flattened block dimension
            self.input_dim = self.feature_dim  # Individual feature dimension
            self.output_dim = self.block_dim   # Full block output
        else:
            raise NotImplementedError(f"Loss function {self.conf.optimizer.loss_function} not supported.")

        if (self.conf.model.extra_future_tokens > 1 or self.conf.model.future_conditioning) and self.conf.model.reduction_factor > 1:
            raise ValueError("extra_future_tokens > 1 is not supported when reduction_factor > 1.")

        # Initialize auxiliary output dimensions for token prediction
        if self.conf.model.ssl_model == "mimi" and self.conf.optimizer.token_loss_weight > 0:
            n_special_tokens = getattr(self.conf.model, "n_special_tokens", 0)
            self.aux_output_dim = self.ssl_model.model.config.codebook_size + n_special_tokens
            # hardcoded: use the last index as eos token
            self.eos_token_index = self.aux_output_dim - 1
            if hasattr(self.conf.model, "extra_future_tokens") and self.conf.model.extra_future_tokens > 0:
                self.aux_output_dim = self.aux_output_dim * (self.conf.model.extra_future_tokens * reduction_factor)
        else:
            self.aux_output_dim = None

        # Initialize token embedding dimensions
        self.token_emb_dim = self.conf.model.token_emb_dim if hasattr(self.conf.model, "token_emb_dim") and hasattr(self.conf.model, "token_conditioning") and self.conf.model.token_conditioning else 0
        if hasattr(self.conf.model, "future_conditioning") and self.conf.model.future_conditioning:
            self.token_emb_dim *= self.conf.model.extra_future_tokens
        self.token_emb_dim = self.token_emb_dim * reduction_factor

        # print("DEBUG", self.conf.model.decoder)
        if "OpenELM" in self.conf.model.decoder:
            output_layer = "simple_mlp" if self.conf.optimizer.loss_function == "FM" else "linear"
            self._output_layer_type = output_layer
            self._n_res_blocks = self.conf.model.n_res_blocks
            self.aux_output_layer_idx = None if not hasattr(self.conf.model, "aux_output_layer_idx") else self.conf.model.aux_output_layer_idx

            if hasattr(self.conf.model, "ssl_model") and self.conf.model.ssl_model == "mimi":
                self.decoder = ELMBlockDecoderWrapper(
                    decoder_model,
                    input_dim=self.feature_dim,
                    decoder_dim=self.conf.model.decoder_dim,
                    output_dim=self.conf.model.decoder_dim,  # Output for conditioning
                    block_size=self.block_size,
                    aux_output_dim=self.aux_output_dim,
                    output_layer=self._output_layer_type,
                    n_res_blocks=self._n_res_blocks,
                    aux_output_layer_idx=self.aux_output_layer_idx,
                    token_emb_dim=self.token_emb_dim,
                )
            # use self._lm for config access
            self.pad_index = getattr(decoder_model.config, "pad_token_id", decoder_model.config.bos_token_id) #!!!!
            self.bos_index = decoder_model.config.bos_token_id
            self.eos_index = decoder_model.config.eos_token_id

    def _init_embeddings(self):
        """Create embedding layers for block modeling."""
        # Initialize embeddings for block conditioning (no BOS for block model)
        self.null_block = nn.Parameter(torch.randn(self.block_dim) * 0.02)

        # Initialize token embeddings if needed
        if hasattr(self.conf.model, "token_conditioning") and self.conf.model.token_conditioning:
            # add token emb to z, only support mimi
            if hasattr(self.conf.model, "add_special_token_to_embedding_table") and self.conf.model.add_special_token_to_embedding_table:
                self.token_embed = nn.Embedding(self.ssl_model.model.config.codebook_size + self.conf.model.n_special_tokens, embedding_dim=self.conf.model.token_emb_dim)
            else:
                self.token_embed = nn.Embedding(self.ssl_model.model.config.codebook_size, embedding_dim=self.conf.model.token_emb_dim)
    
    def _split_into_blocks(self, ssl_feats: torch.Tensor, wav_len: torch.Tensor):
        """Split SSL features into blocks for joint modeling.

        Args:
            ssl_feats: [B, T, F] SSL features
            wav_len: [B] sequence lengths

        Returns:
            blocks: [B, num_blocks, block_size, F] blocked features
            block_mask: [B, num_blocks] mask for valid blocks
        """
        B, T, F = ssl_feats.shape

        # Pad sequence to be divisible by block_size
        num_blocks = (T + self.block_size - 1) // self.block_size
        padded_T = num_blocks * self.block_size

        if padded_T > T:
            padding = torch.zeros(B, padded_T - T, F, device=ssl_feats.device, dtype=ssl_feats.dtype)
            ssl_feats = torch.cat([ssl_feats, padding], dim=1)

        # Reshape into blocks: [B, T, F] → [B, num_blocks, block_size, F]
        blocks = ssl_feats.view(B, num_blocks, self.block_size, F)

        # Create block mask based on sequence lengths
        block_lengths = torch.ceil(wav_len * T / self.block_size).long()
        block_mask = torch.arange(num_blocks, device=ssl_feats.device)[None, :] < block_lengths[:, None]

        return blocks, block_mask

    def _get_ssl_feats(self, wavs, wav_len):
        with torch.no_grad():
            if self.conf.model.ssl_model == "mimi" and hasattr(self.conf.model, "n_quantizers") and self.conf.model.n_quantizers > 0:
                ssl_feats, tokens = self.ssl_model(wavs, wav_len)
            else:
                raise NotImplementedError(f"SSL model {self.conf.model.ssl_model} not supported.")

            ssl_abs_len = torch.round(wav_len * ssl_feats.shape[1]).long()
            #ssl_padding_mask = ~length_to_mask(ssl_abs_len, dtype=torch.bool)
            
            if hasattr(self.conf.model, "norm") and self.conf.model.norm == "static":
                ssl_feats = (ssl_feats - self.mean) / self.std

            # Reduce features
            if self.conf.model.reduction_factor > 1:
                reduced_ssl_feats = reduce_features(ssl_feats, self.conf.model.reduction_factor, pad=False)
            else:
                reduced_ssl_feats = ssl_feats
        return reduced_ssl_feats, ssl_feats, ssl_abs_len, tokens

    def _process_token_predictions(self, aux_output, wav_len, tokens, bs):
        """Extracted logic for processing token predictions.

        Returns (token_logits, tokens, split_padding_mask).
        If tokens is None or token prediction is disabled, returns (None, None, None).
        """
        # If tokens is None, nothing to do
        if tokens is None:
            return None, None, None

        if self.conf.model.ssl_model == "mimi" and (
            self.conf.optimizer.token_loss_weight > 0 or self.conf.model.token_conditioning
        ) and self.conf.model.n_quantizers > 0:
            token_logits = split_features(aux_output, self.conf.model.reduction_factor)  # [B, T * r, F // r]
            # aux_output is block-level [B, num_blocks, aux_dim]; expand to frame-level
            token_logits = token_logits.repeat_interleave(self.block_size, dim=1)  # [B, num_blocks * block_size, aux_dim]
            k = 1 if not hasattr(self.conf.model, "extra_future_tokens") or self.conf.model.extra_future_tokens == 0 else self.conf.model.extra_future_tokens
            ssl_abs_len = torch.round(wav_len * tokens.shape[1]).long()
            # add one for eos token
            split_padding_mask = length_to_mask(ssl_abs_len + 1, dtype=torch.bool)
            # append eos as last k tokens 
            eos_index = self.eos_token_index
            tokens = torch.cat([tokens, tokens.new_ones((bs, k, 1)).long() * eos_index], dim=1)  # shape [B, T + k, 1]
            offsets = ssl_abs_len.unsqueeze(1) + torch.arange(k, device=tokens.device).unsqueeze(0)  # shape [B, k]
            batch_indices = torch.arange(bs, device=tokens.device).unsqueeze(1).expand(bs, k)         # shape [B, k]
            tokens[batch_indices, offsets, 0] = eos_index
            token_logits = token_logits[:, :split_padding_mask.shape[1], :]  # align to loss mask length
        else:
            token_logits = None
            tokens = None
            split_padding_mask = None

        return token_logits, tokens, split_padding_mask

    def _apply_token_conditioning_and_padding(self, logits, tokens, padding_mask, abs_len, bs):
        """Apply token conditioning to logits (if enabled) and compute padding_mask_for_loss.

        Returns (logits, padding_mask_for_loss).
        """
        # Apply token conditioning if specified
        if hasattr(self.conf.model, "token_conditioning") and self.conf.model.token_conditioning:
            L = logits.shape[1]
            if hasattr(self.conf.model, "future_conditioning") and self.conf.model.future_conditioning:
                k = 1 if self.conf.model.extra_future_tokens == 0 else self.conf.model.extra_future_tokens
                conditioning_tokens = torch.stack([tokens[:, kk:kk+L, 0] for kk in range(k)], dim=2) # [B, T, k]
                token_embed = self.token_embed(conditioning_tokens).flatten(start_dim=2, end_dim=-1) # [B, T, k * D]
                logits = torch.cat([logits, token_embed], dim=2)
            elif self.conf.model.reduction_factor > 1:
                token_embed = self.token_embed(tokens[:, :L * self.conf.model.reduction_factor, 0])
                token_embed = reduce_features(token_embed, self.conf.model.reduction_factor, pad=False)
                logits = torch.cat([logits, token_embed], dim=2)
            elif self.conf.model.reduction_factor == 1:
                # use only the first token
                token_embed = self.token_embed(tokens[:, :L, 0])
                logits = torch.cat([logits, token_embed], dim=2)

        # Remove one frame for loss computation
        padding_mask_for_loss = padding_mask.clone()
        padding_mask_for_loss[torch.arange(bs, device=padding_mask_for_loss.device), abs_len - 1] = 0
        padding_mask_for_loss = padding_mask_for_loss[:, :-1].unsqueeze(dim=2)

        return logits, padding_mask_for_loss

    def forward(self, wavs, wav_len, **decoder_kwargs):
        """Forward pass for block diffusion pipeline.

        Returns:
            block_reprs: [B, num_blocks, decoder_dim] LM representations per block
            target_blocks: [B, num_blocks, block_dim] flattened target blocks
            block_mask: [B, num_blocks] valid block mask
            token_logits: auxiliary token predictions (if enabled)
            tokens: quantized tokens (if available)
            split_padding_mask: token mask (if available)
        """
        reduced_ssl_feats, ssl_feats, ssl_abs_len, tokens = self._get_ssl_feats(wavs, wav_len)

        # Split features into blocks for joint modeling
        blocks, block_mask = self._split_into_blocks(reduced_ssl_feats, wav_len)
        B, num_blocks, block_size, F = blocks.shape

        # Flatten each block: [B, num_blocks, block_size, F] → [B, num_blocks, block_size * F]
        target_blocks = blocks.reshape(B, num_blocks, self.block_dim)

        # Process blocks autoregressively through LM
        block_sequence = target_blocks.clone()  # [B, num_blocks, block_dim]

        # Get LM representations for each block (conditioned on previous blocks)
        block_reprs, aux_output = self.decoder(block_sequence, attention_mask=block_mask)

        # Process auxiliary token predictions if needed
        bs = B
        token_logits, tokens, split_padding_mask = self._process_token_predictions(aux_output, wav_len, tokens, bs)

        return block_reprs, target_blocks, block_mask, token_logits, tokens, split_padding_mask