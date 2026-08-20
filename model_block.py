from typing import Optional, Tuple, Union, List
import math
import logging

import torch
import torch.nn as nn
from contextlib import nullcontext

from transformers import MimiModel, AutoFeatureExtractor

try:
    from model_utils import modulate
except ImportError:
    def modulate(x, shift, scale):
        return x * (1 + scale) + shift

logger = logging.getLogger(__name__)

MIMI_REVISION = "89091b3e466eb6a9d11e537bf26b144f194978f7"


class MimiEncoder(torch.nn.Module):
    """Mimi encoder for speech representation learning."""

    def __init__(self, freeze: bool = True, n_quantizers: int = 0):
        super().__init__()
        self.model = MimiModel.from_pretrained("kyutai/mimi", revision=MIMI_REVISION)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "kyutai/mimi", revision=MIMI_REVISION
        )
        self.freeze = freeze
        self.n_quantizers = n_quantizers

        if freeze:
            self.model.eval()
            # Freeze parameters
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, wavs: torch.Tensor, wav_lens: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        context = torch.no_grad() if self.freeze else nullcontext()
        with context:
            embeddings = self.model.encoder(wavs.unsqueeze(dim=1))
            encoder_outputs = self.model.encoder_transformer(
                embeddings.transpose(1, 2), past_key_values=None, return_dict=None
            )
            embeddings = encoder_outputs[0].transpose(1, 2)
            embeddings = self.model.downsample(embeddings)

        if self.n_quantizers > 0:
            codes = self.model.quantizer.encode(embeddings, self.n_quantizers)
            codes = codes.transpose(0, 1)
            return embeddings.transpose(1, 2), codes.transpose(1, 2)  # [B, T, F], [B, T, C]
        else:
            return embeddings.transpose(1, 2)


class MimiDecoder(torch.nn.Module):
    """Mimi decoder for speech synthesis."""

    def __init__(self):
        super().__init__()
        self.model = MimiModel.from_pretrained("kyutai/mimi", revision=MIMI_REVISION)

    def forward(self, embeddings: torch.Tensor, num_quantizers: Optional[int] = None, return_codes: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        num_quantizers = self.model.config.num_quantizers if num_quantizers is None else num_quantizers
        embeddings = embeddings.transpose(1, 2)
        codes = self.model.quantizer.encode(embeddings, num_quantizers)
        codes = codes.transpose(0, 1)
        audio_values = self.model.decode(codes)[0].squeeze(dim=1)
        if not return_codes:
            return audio_values
        else:
            return audio_values, codes


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000, scale: float = 1000.0) -> torch.Tensor:
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=t.dtype) / half
        ).to(device=t.device)
        args = t[:, :, None].float() * scale * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """A residual block with adaptive layer normalization."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """The final layer adopted from DiT."""

    def __init__(self, model_channels: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class BlockFlowNet(nn.Module):
    """Flow matching network for joint block denoising.

    Processes a block of acoustic frames as a sequence [B, block_size, feature_dim]
    using self-attention for within-block temporal modeling, then applies per-frame
    ResBlocks conditioned on the diffusion timestep and LM representation.
    """

    def __init__(
        self,
        block_size: int,       # number of acoustic frames per block
        feature_dim: int,      # dimension of each acoustic frame
        model_channels: int,
        z_channels: int,       # conditioning dimension from LM
        num_res_blocks: int,
        num_attn_heads: int = 8,
        grad_checkpointing: bool = False
    ):
        super().__init__()

        self.block_size = block_size
        self.feature_dim = feature_dim
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        # Project each frame independently to model dimension
        self.frame_proj = nn.Linear(feature_dim, model_channels)

        # Self-attention across frames within the block for temporal modeling
        self.attn_norm = nn.LayerNorm(model_channels)
        self.attn = nn.MultiheadAttention(model_channels, num_heads=num_attn_heads, batch_first=True)

        # Per-frame ResBlocks with adaptive conditioning from t + z
        self.res_blocks = nn.ModuleList([ResBlock(model_channels) for _ in range(num_res_blocks)])

        # Per-frame output projection back to feature_dim
        self.final_layer = FinalLayer(model_channels, feature_dim)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: [B, block_size, feature_dim]
        # t: [B]
        # c: [B, z_channels]
        weight_dtype = self.frame_proj.weight.dtype
        x = x.to(weight_dtype)
        t = t.to(weight_dtype)
        c = c.to(weight_dtype)

        x = self.frame_proj(x)                           # [B, block_size, model_channels]
        t_emb = self.time_embed(t.unsqueeze(1)).squeeze(1)  # [B, model_channels]
        c_emb = self.cond_embed(c)                        # [B, model_channels]
        y = t_emb + c_emb                                 # [B, model_channels]

        # Self-attention across frames to capture within-block temporal dependencies
        x_norm = self.attn_norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out                                  # [B, block_size, model_channels]

        # Per-frame ResBlocks; y is broadcast via unsqueeze: [B, 1, model_channels]
        for block in self.res_blocks:
            x = block(x, y.unsqueeze(1))

        # Per-frame output: [B, block_size, feature_dim]
        out = self.final_layer(x, y.unsqueeze(1))
        return out


class BaseDecoderWrapper(torch.nn.Module):
    """Base class for decoder wrappers."""

    def __init__(
        self,
        model,
        input_dim: int,
        decoder_dim: int,
        output_dim: int,
        aux_output_dim: Optional[int] = None,
        output_layer: str = "linear",
        n_res_blocks: int = 3,
        aux_output_layer_idx: Optional[int] = None,
        token_emb_dim: int = 0
    ):
        super().__init__()
        self.input_proj = torch.nn.Linear(input_dim, decoder_dim)
        self.aux_output_layer_idx = aux_output_layer_idx
        self.output_layer_type = output_layer
        self.frozen = False

        # Initialize output projection
        if output_layer == "linear":
            self.output_proj = torch.nn.Linear(decoder_dim, output_dim)
        elif output_layer == "simple_mlp":
            if decoder_dim > 1280:
                self.output_proj = SimpleMLPAdaLN(output_dim, decoder_dim, output_dim, decoder_dim + token_emb_dim, n_res_blocks)
            else:
                self.output_proj = SimpleMLPAdaLN(output_dim, decoder_dim * 2, output_dim, decoder_dim + token_emb_dim, n_res_blocks)

        if aux_output_dim:
            self.aux_output_proj = torch.nn.Linear(decoder_dim, aux_output_dim)


class ELMBlockDecoderWrapper(BaseDecoderWrapper):
    """Block-aware decoder wrapper for OpenELM models with joint block modeling."""

    def __init__(
        self,
        elm,
        input_dim: int,
        decoder_dim: int,
        output_dim: int,
        block_size: int,
        aux_output_dim: Optional[int] = None,
        output_layer: str = "linear",
        n_res_blocks: int = 3,
        aux_output_layer_idx: Optional[int] = None,
        token_emb_dim: int = 0
    ):
        # For block modeling, input_dim should be block_size * feature_dim
        super().__init__(elm, input_dim, decoder_dim, output_dim, aux_output_dim, output_layer, n_res_blocks, aux_output_layer_idx, token_emb_dim)
        self.decoder = elm.transformer
        self.block_size = block_size

        # Block projection layer: flatten block → decoder_dim
        self.block_proj = nn.Linear(input_dim * block_size, decoder_dim)
        

    def _create_block_attention_mask(self, num_blocks: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Create attention mask for block processing: causal between blocks, full within blocks."""
        # Create mask: [num_blocks, num_blocks] where entry (i,j) = can block i attend to block j?
        mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device, dtype=dtype))
        return mask

    def forward(
        self,
        block_sequence: torch.Tensor,  # [B, num_blocks, block_size * feature_dim]
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        batch_size, num_blocks, block_dim = block_sequence.shape

        # Project each flattened block to decoder dimension
        # [B, num_blocks, block_size * feature_dim] → [B, num_blocks, decoder_dim]
        inputs_embeds = self.block_proj(block_sequence)

        # Create position IDs for blocks (not individual tokens)
        cache_position = torch.arange(0, num_blocks, device=block_sequence.device)
        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

        # Create block-level causal mask: causal between blocks
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, num_blocks, device=block_sequence.device, dtype=torch.bool)

        # Update causal mask for block-level attention
        causal_mask = self.decoder._update_causal_mask(attention_mask, inputs_embeds)

        hidden_states = inputs_embeds

        # Process through transformer layers
        for idx, decoder_layer in enumerate(self.decoder.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=None,
                use_cache=None,
                cache_position=cache_position,
            )

            if self.aux_output_layer_idx is not None and idx == self.aux_output_layer_idx - 1:
                aux_hidden_states = layer_outputs[0]
            hidden_states = layer_outputs[0]

            if torch.isnan(hidden_states).any():
                print("\n--- NaN Detected in Block Processing ---")
                raise RuntimeError("NaN in block processing")

        if self.aux_output_layer_idx is None:
            aux_hidden_states = hidden_states

        hidden_states = self.decoder.norm(hidden_states)

        # For block diffusion, we return block-level representations
        # These will be used by the flow matching head for joint block prediction
        if self.output_layer_type == "simple_mlp":
            logits = hidden_states  # [B, num_blocks, decoder_dim]
        elif self.output_layer_type == "linear":
            logits = self.output_proj(hidden_states)
        else:
            raise ValueError(f"output_layer {self.output_layer_type} not supported")

        if hasattr(self, "aux_output_proj"):
            aux_output = self.aux_output_proj(aux_hidden_states)
        else:
            aux_output = None

        return logits, aux_output

# from spidr.models.spidr import SpidR
# from spidr.config import SpidRConfig
# from dataclasses import replace

# class SPIDREncoder(torch.nn.Module):
#     def __init__(self, conf, freeze=True):
#         super().__init__()
#         spidr_cfg = SpidRConfig()
#         spidr_cfg = replace(
#             spidr_cfg,
#             extractor_mode="layer_norm"
#         )
#         self.model = SpidR(spidr_cfg)

#         for p in self.model.parameters():
#             p.requires_grad = False

#         self.model.eval()
        
#         self.freeze = freeze
#         self.model.config = spidr_cfg

#         # projection to match Mimi dim if needed
#         self.spidr_dim = self.model.config.encoder_embed_dim
#         if self.spidr_dim != conf.model.ssl_dim:
#             self.proj = nn.Linear(self.spidr_dim, conf.model.ssl_dim)
#         else:
#             self.proj = nn.Identity()

#         if freeze:
#             self.model.eval()
#             for p in self.model.parameters():
#                 p.requires_grad = False

#     def forward(self, wavs, wav_lens):
#         context = torch.no_grad() if self.freeze else nullcontext()
#         with context:
#             codebooks = self.model.get_codebooks(wavs, onehot=False)

#         # last codebook is the semantic one
#         tokens = codebooks[-1]              # [B, T, codebook_size]
#         tokens = tokens.argmax(-1)          # [B, T]
#         tokens = tokens.unsqueeze(-1)       # [B, T, 1]

#         feats = None  # we don't use SPIDR features
#         return feats, tokens
