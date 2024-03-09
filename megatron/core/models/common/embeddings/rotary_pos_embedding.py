# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.transformer_block import TransformerBlock

import logging

import torch
from torch import Tensor, nn
from nemo.utils import logging
from typing import Dict, Any
import random
import math

from megatron.core import parallel_state

logger = logging.getLogger(__name__)

try:
    from apex.transformer.functional import (
        fused_apply_rotary_pos_emb,
        fused_apply_rotary_pos_emb_thd,
    )

    HAVE_APPLY_ROPE_FUSION = True
except:
    HAVE_APPLY_ROPE_FUSION = False


__all__ = ['RotaryEmbedding', 'apply_rotary_pos_emb']


def get_pos_emb_on_this_cp_rank(pos_emb, seq_dim):
    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()
    cp_idx = torch.tensor(
        [cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True
    ).cuda(non_blocking=True)
    pos_emb = pos_emb.view(
        *pos_emb.shape[:seq_dim], 2 * cp_size, -1, *pos_emb.shape[(seq_dim + 1) :]
    )
    pos_emb = pos_emb.index_select(seq_dim, cp_idx)
    pos_emb = pos_emb.view(*pos_emb.shape[:seq_dim], -1, *pos_emb.shape[(seq_dim + 2) :])
    return pos_emb


class RotaryEmbedding(nn.Module):
    """Rotary Embedding for language model.

    Args:
        kv_channels (int): Projection weights dimension in multi-head attention. Obtained from transformer config
        rotary_percent (float): Percent of rotary dimension to use for rotary position embeddings.
        seq_len_interpolation_factor (float, optional): scale of linearly interpolating RoPE for longer sequences. The value must be a float larger than 1.0. Defaults to None
        rotary_base (int, optional): Base period for rotary position embeddings. Defaults to 10000.
    """

    def __init__(
        self,
        kv_channels: int,
        rotary_percent: float,
        rotary_interleaved: bool = False,
        seq_len_interpolation_factor: float = None,
        rotary_base: int = 10000,
        pretrained_max_position_embeddings: int = 4096,
        augment_seq: Dict[Any,Any] = None,
        logging_freq: int = 0.01,
    ) -> None:
        super().__init__()

        dim = kv_channels
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)
        self.rotary_interleaved = rotary_interleaved

        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        if augment_seq and 'wavelengths' in augment_seq:
            wavelengths = torch.tensor(augment_seq['wavelengths'], dtype=torch.float32, device=torch.cuda.current_device())
            self.inv_freq = 2 * math.pi / wavelengths
            logging.info(f'using passed in wavelengths {augment_seq['wavelengths']}')
        else:
            self.inv_freq = 1.0 / (
                rotary_base
                ** (
                    torch.arange(0, dim, 2, dtype=torch.float32, device=torch.cuda.current_device())
                    / dim
                )
            )
        self.pretrained_max_position_embeddings = pretrained_max_position_embeddings
        self.augment_seq = augment_seq
        self.logging_freq = logging_freq
        logging.info(f'pretrained_max_position_embeddings: {pretrained_max_position_embeddings}, rotary_base: {rotary_base}, seq_len_interpolation_factor: {seq_len_interpolation_factor}, augment_seq: {augment_seq}')

    """
        Augments the seq and adjusts its range to base_len
        Args:
            seq (tensor): tensor of positions
            max_seq_len (int): length of this samplw
            Applies stretch and shift augmentations and returns the augmented seq
    """
    def augment(self, seq, max_seq_len):
        current_range = max_seq_len

        target_augmented_length = self.augment_seq.get('target', None)
        augmented_length_range = self.augment_seq.get('range', None)
        if target_augmented_length and augmented_length_range:
            logging.warning(f'target_augmented_length setting of {target_augmented_length} supercedes augmented_length_range of {augmented_length_range}')
        elif augmented_length_range:
            target_augmented_length = random.randint(max(augmented_length_range[0], max_seq_len),augmented_length_range[1])

        if self.augment_seq.get('stretch', False):
            if target_augmented_length:
                max_stretch_factor  = target_augmented_length / current_range
            else:
                max_stretch_factor  = self.base_len * self.seq_len_interpolation_factor / current_range

            stretch_factor = random.random() * max_stretch_factor
            if self.augment_seq.get('discrete', False):
                stretch_factor = int(stretch_factor)
            seq *= stretch_factor
            current_range *= stretch_factor
        
        num_shifts = self.augment_seq.get('num_shifts', None)
        if num_shifts:
            if target_augmented_length:
                total_shift = target_augmented_length - current_range
            else:
                total_shift = self.base_len * self.seq_len_interpolation_factor - current_range

        if self.augment_seq.get('allowed_shift_values', False):
            # provides allowed values for each shift index
            allowed_shift_values = self.augment_seq['allowed_shift_values']
            assert (len(allowed_shift_values) == num_shifts), f'allowed_shift_values length {allowed_shift_values} does not match num_shifts {num_shifts}'
            shifts = torch.zeros(num_shifts, dtype = torch.int)
            for idx, allowed_values in enumerate(allowed_shift_values):
                shifts[idx] = random.choice(allowed_values)
                
        else:
            shifts = torch.rand(num_shifts)
            if augmented_length_range is not None:
                shifts = (augmented_length_range[0] + shifts * (augmented_length_range[1] - augmented_length_range[0]))/ num_shifts
            else:
                shifts = shifts / shifts.sum() * total_shift
            
            if self.augment_seq.get('discrete', False):
                shifts = torch.round(shifts).to(torch.int)

        if self.augment_seq.get('shift_indices', False):
            indices2shift = self.augment_seq['shift_indices']
        else:
            indices2shift = (torch.rand(num_shifts) * max_seq_len).to(torch.int)

        for idx, i in enumerate(indices2shift):
            seq[i:] += shifts[idx]

        if random.random() < self.logging_freq:
            logging.info(f'indices2shift: {indices2shift}, shifts: {shifts}, total shift: {torch.sum(shifts)}')

        return seq
        

    def forward(self, max_seq_len: int, offset: int =0, maybe_augment: bool=True) -> Tensor:

        """Forward pass of RoPE embedding.

        Args:
            max_seq_len (int): Maximum size of sequence
            offset (int, optional): _description_. Defaults to 0.

        Returns:
            Tensor: Embeddings after applying RoPE.
        """

        if random.random() < self.logging_freq:
            logging.info(f'max_seq_len: {max_seq_len}, maybe_augment: {maybe_augment}')

        seq = (
            torch.arange(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            + offset
        )

        if max_seq_len > self.pretrained_max_position_embeddings * self.seq_len_interpolation_factor:
            # dynamic linear scaling (length > position we have learned)
            logging.info(f'dynamic interpolation triggered: max_seq_len: {max_seq_len}, pretrained_max_position_embeddings: {self.pretrained_max_position_embeddings}, seq_len_interpolation_factor: {self.seq_len_interpolation_factor}')
            seq *= 1 / (max_seq_len / self.pretrained_max_position_embeddings)
        else:
            if maybe_augment and self.augment_seq:
                seq = self.augment(seq, max_seq_len)

            if self.seq_len_interpolation_factor is not None:
                seq *= 1 / self.seq_len_interpolation_factor

        freqs = torch.outer(seq, self.inv_freq)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        if not self.rotary_interleaved:
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            emb = torch.stack((freqs.view(-1, 1), freqs.view(-1, 1)), dim=-1).view(
                freqs.shape[0], -1
            )
        # emb [seq_length, .., dim]
        emb = emb[:, None, None, :]
        if parallel_state.get_context_parallel_world_size() > 1:
            # slice rotary_pos_emb along sequence dimension and select the parition of the current CP rank
            emb = get_pos_emb_on_this_cp_rank(emb, 0)
        return emb

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        state_dict.pop(f'{prefix}inv_freq', None)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def get_rotary_seq_len(
        self,
        inference_params,
        transformer: TransformerBlock,
        transformer_input: Tensor,
        transformer_config: TransformerConfig,
    ) -> float:
        """Function to get the rotary sequence length.

        Args:
            inference_params : Used during Inference time
            transformer (TransformerBlock): The transformer block (decoder/encoder) used by the model
            transformer_input (Tensor): _description_
            transformer_config (TransformerConfig): Transformer config used by the model

        Returns:
            float: The rotary sequence length
        """
        if inference_params is not None:
            rotary_seq_len = inference_params.max_sequence_length
        else:
            if transformer.input_tensor is not None:
                rotary_seq_len = transformer.input_tensor.size(0)
            else:
                rotary_seq_len = transformer_input.size(0)

            if transformer_config.sequence_parallel:
                rotary_seq_len *= transformer_config.tensor_model_parallel_size

        rotary_seq_len *= transformer_config.context_parallel_size

        return rotary_seq_len


def _rotate_half(x: Tensor, rotary_interleaved: bool) -> Tensor:
    """Change sign so the last dimension becomes [-odd, +even]

    Args:
        x (Tensor): Input tensor

    Returns:
        Tensor: Tensor rotated half
    """
    if not rotary_interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1 = x[:, :, :, ::2]
        x2 = x[:, :, :, 1::2]
        x_new = torch.stack((-x2, x1), dim=-1)
        return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)


def apply_rotary_pos_emb_bshd(t: Tensor, freqs: Tensor, rotary_interleaved: bool = False) -> Tensor:
    """Apply rotary positional embedding to input tensor T.

    check https://kexue.fm/archives/8265 for detailed formulas

    Args:
        t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [seq_length, ..., dim]

    Returns:
        Tensor: The input tensor after applying RoPE
    """
    rot_dim = freqs.shape[-1]

    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)

    t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)
    return torch.cat((t, t_pass), dim=-1)


def apply_rotary_pos_emb_thd(
    t: Tensor, cu_seqlens: Tensor, freqs: Tensor, rotary_interleaved: bool = False
) -> Tensor:

    """A baseline implementation of applying RoPE for `thd` format.

    Args:
        t (Tensor): Input tensor T is of shape [t, h, d]
        cu_seqlens(Tensor):  Cumulative sum of sequence lengths in a batch for `t`,
        with shape [b + 1] and dtype torch.int32.
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [max_s, 1, 1, d]

    Returns:
        Tensor: Shape [t, h, d]. The input tensor after applying RoPE.
    """

    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    return torch.cat(
        [
            apply_rotary_pos_emb_bshd(x.unsqueeze(1), freqs[: x.size(0)])
            for x in torch.split(t, seqlens)
        ]
    ).squeeze(1)


def apply_rotary_pos_emb(
    t: Tensor, freqs: Tensor, config: TransformerConfig, cu_seqlens: Optional[Tensor] = None,
):
    """
    Reroute to the appropriate apply_rotary_pos_emb function depending on
    fused/unfused kernels, or bshd (conventional) / thd (packed seq) format
    """
    if config.apply_rope_fusion and not HAVE_APPLY_ROPE_FUSION:
        # setting apply_rope_fusion in config to False so that subsequent queries to this config also return False
        config.apply_rope_fusion = False
        if not getattr(apply_rotary_pos_emb, "printed_fused_warning", False):
            logger.warning(
                "Setting apply_rope_fusion to false because its implementation"
                " is not included in Apex. Try upgrading to the latest version"
            )
            apply_rotary_pos_emb.printed_fused_warning = True
    if config.apply_rope_fusion:
        if cu_seqlens is None:
            return fused_apply_rotary_pos_emb(t, freqs, transpose_output_memory=True)
        else:
            return fused_apply_rotary_pos_emb_thd(t, cu_seqlens, freqs)
    else:
        if cu_seqlens is None:
            return apply_rotary_pos_emb_bshd(t, freqs, rotary_interleaved=config.rotary_interleaved)
        else:
            return apply_rotary_pos_emb_thd(
                t, cu_seqlens, freqs, rotary_interleaved=config.rotary_interleaved
            )
