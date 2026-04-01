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


class MimiEncoder(torch.nn.Module):
    """Mimi encoder for speech representation learning."""

    def __init__(self, freeze: bool = True, n_quantizers: int = 0):
        super().__init__()
        self.model = MimiModel.from_pretrained("kyutai/mimi")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
        self.freeze = freeze
        self.n_quantizers = n_quantizers

        if freeze:
            self.model.eval()
            # Freeze parameters
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, wavs: torch.Tensor, wav_lens: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Extract Mimi features from input waveform.

        Args:
            wavs: Input waveform tensor
            wav_lens: Waveform length tensor

        Returns:
            Extracted features tensor, optionally with quantized codes
        """
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
        self.model = MimiModel.from_pretrained("kyutai/mimi")

    def forward(self, embeddings: torch.Tensor, num_quantizers: Optional[int] = None, return_codes: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Decode Mimi embeddings to audio.

        Args:
            embeddings: Input embeddings tensor
            num_quantizers: Number of quantizers to use
            return_codes: Whether to return quantized codes

        Returns:
            Decoded audio tensor, optionally with codes
        """
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
        """Create sinusoidal timestep embeddings.

        Args:
            t: A 1-D Tensor of N indices, one per batch element. These may be fractional.
            dim: The dimension of the output.
            max_period: Controls the minimum frequency of the embeddings.
            scale: Scaling factor for the embeddings.

        Returns:
            An (N, D) Tensor of positional embeddings.
        """
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
        """Forward pass for timestep embedding.

        Args:
            t: Timestep tensor

        Returns:
            Embedded timestep tensor
        """
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
        """Forward pass for residual block with adaptive layer norm.

        Args:
            x: Input tensor
            y: Conditioning tensor

        Returns:
            Output tensor with residual connection
        """
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
        """Forward pass for final layer.

        Args:
            x: Input tensor
            c: Conditioning tensor

        Returns:
            Output tensor
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class BlockFlowNet(nn.Module):
    """Flow matching network for joint block denoising with adaptive layer normalization."""

    def __init__(
        self,
        block_dim: int,  # block_size * feature_dim (flattened block dimension)
        model_channels: int,
        z_channels: int,  # conditioning dimension from LM
        num_res_blocks: int,
        grad_checkpointing: bool = False
    ):
        super().__init__()

        self.block_dim = block_dim
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        # Project flattened noisy block to model dimension
        self.input_proj = nn.Linear(block_dim, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(model_channels))

        self.res_blocks = nn.ModuleList(res_blocks)

        # Output back to flattened block space for joint prediction
        self.final_layer = FinalLayer(model_channels, block_dim)

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize model weights."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Apply block flow network to noisy flattened block.

        Args:
            x: [B, block_dim] flattened noisy block
            t: [B] timestep for each batch element
            c: [B, z_channels] conditioning from LM for this block

        Returns:
            [B, block_dim] predicted velocity/noise for joint block
        """
        x = self.input_proj(x)  # [B, block_dim] → [B, model_channels]
        t = self.time_embed(t.unsqueeze(1))  # [B] → [B, 1, model_channels]
        t = t.squeeze(1)  # [B, model_channels]
        c = self.cond_embed(c)  # [B, z_channels] → [B, model_channels]

        y = t + c  # [B, model_channels]

        if self.grad_checkpointing and not torch.jit.is_scripting():
            from torch.utils.checkpoint import checkpoint
            for block in self.res_blocks:
                x = checkpoint(block, x, y.unsqueeze(1))
                x = x.squeeze(1)
        else:
            for block in self.res_blocks:
                x = block(x.unsqueeze(1), y.unsqueeze(1))  # Add sequence dim for ResBlock
                x = x.squeeze(1)  # Remove sequence dim

        out = self.final_layer(x.unsqueeze(1), y.unsqueeze(1))  # Add sequence dim
        out = out.squeeze(1)  # [B, block_dim]
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
        """Forward pass for block-aware ELM decoder.

        Args:
            block_sequence: [B, num_blocks, block_size * feature_dim] flattened blocks
            attention_mask: Optional mask for sequence length

        Returns:
            Block-level representations for flow matching
        """
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

