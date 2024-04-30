# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.transformer_block import TransformerBlock

import torch
from torch import Tensor, nn
from nemo.utils import logging
from typing import Dict, Any
import random
import math

from megatron.core import parallel_state

__all__ = ['RotaryEmbedding', 'apply_rotary_pos_emb']


def get_pos_emb_on_this_cp_rank(pos_emb, seq_dim):
    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()
    cp_idx = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device=pos_emb.device)
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
        seq_len_interpolation_factor: float = None,
        rotary_base: int = 10000,
        pretrained_max_position_embeddings: int = 4096,
        augment_seq: Dict[Any,Any] = None,
        logging_freq: int = 0.01,
    ) -> None:
        super().__init__()

        self.rotary_base = rotary_base
        dim = kv_channels
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)

        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        if augment_seq and 'wavelengths' in augment_seq:
            self.inv_freq = 2 * math.pi / torch.tensor(augment_seq['wavelengths'], dtype=torch.float32, device=torch.cuda.current_device())
            logging.info(f'using passed in wavelengths {augment_seq["wavelengths"]}')
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

        logging.info(f'kv_channels: {kv_channels}, rotary_percent: {rotary_percent}')
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
        if num_shifts is not None and num_shifts > 0:
            if target_augmented_length:
                total_shift = target_augmented_length - current_range
            else:
                total_shift = self.base_len * self.seq_len_interpolation_factor - current_range
        else:
            return seq

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
            unshifted_seq = None
            if maybe_augment and self.augment_seq and random.random() < self.augment_seq.get('freq', 1.0) and max_seq_len > self.augment_seq.get('min_seq_len', 0):
                unshifted_seq = seq.clone()
                seq = self.augment(seq, max_seq_len)

            if self.seq_len_interpolation_factor is not None:
                seq *= 1 / self.seq_len_interpolation_factor
                if unshifted_seq is not None:
                    unshifted_seq *= 1 / self.seq_len_interpolation_factor


        if unshifted_seq is not None and 'token_specific_bases' in self.augment_seq:
            tsb = self.augment_seq['token_specific_bases'] 
            # we start at pos 0 with the default base
            # then we apply the new base[s] from the stated position till the next one or until the end.
            # [10:100000, 20:1000000]
            dim = self.inv_freq.shape[0] * 2
            # token specific inverted frequencies: T x D
            tsif = self.inv_freq.unsqueeze(0).expand(max_seq_len, self.inv_freq.shape[0]).clone()
            previous_tok_cutoff = 0
            previous_base = self.rotary_base
            for tok_cutoff, base in tsb.items(): # careful here. the token cutoffs should not exceed max_seq_len
                  # we apply cutoffs on the shifted positions.so maybe start your shifts not from 0
                if tok_cutoff > max_seq_len:
                    break # do we need to handle the end?
                tsif[previous_tok_cutoff:tok_cutoff, :] = previous_base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=torch.cuda.current_device()) / dim)
                previous_tok_cutoff = tok_cutoff
                previous_base = base
            tsif[previous_tok_cutoff:, :] = previous_base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=torch.cuda.current_device()) / dim)

            freqs = torch.mul(seq.unsqueeze(1), tsif)
            del tsif

        elif unshifted_seq is not None and self.augment_seq.get('min_dim_shifted', None):
            # fseq: T x D
            fseq = unshifted_seq.unsqueeze(1).expand(max_seq_len, self.inv_freq.shape[0]).clone()
            fseq[:,self.augment_seq.get('min_dim_shifted'):] = seq.view(-1,1)
            freqs = torch.mul(fseq, self.inv_freq)
            del fseq
        else:
            freqs = torch.outer(seq, self.inv_freq)

        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        emb = torch.cat((freqs, freqs), dim=-1)
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


def _rotate_half(x: Tensor) -> Tensor:
    """Change sign so the last dimension becomes [-odd, +even]

    Args:
        x (Tensor): Input tensor

    Returns:
        Tensor: Tensor rotated half
    """

    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t: Tensor, freqs: Tensor) -> Tensor:
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

    t = (t * cos_) + (_rotate_half(t) * sin_)
    return torch.cat((t, t_pass), dim=-1)
