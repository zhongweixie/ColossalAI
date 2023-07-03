from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributed import ProcessGroup

from colossalai.shardformer.layer.parallel_module import ParallelModule

__all__ = ['FlashAttentionForOPT']


class FlashAttentionForOPT(ParallelModule):

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()
        # get query proj
        query_states = self._shape(self.q_proj(hidden_states), -1, bsz)
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = self._shape(past_key_value[0].transpose(1, 2).contiguous())
            value_states = self._shape(past_key_value[1].transpose(1, 2).contiguous())
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=1)
            value_states = torch.cat([past_key_value[1], value_states], dim=1)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        src_len = key_states.size(1)
        if layer_head_mask != None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                                 f" {layer_head_mask.size()}")
        if attention_mask != None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}")
            attention_mask = attention_mask.expand(bsz, self.num_heads, tgt_len, tgt_len).contiguous()

        try:
            from xformers.ops import memory_efficient_attention as me_attention
        except ImportError:
            print("Error: xformers module is not installed. Please install it to use flash attention.")
        attn_output = me_attention(query_states,
                                   key_states,
                                   value_states,
                                   attn_bias=attention_mask,
                                   p=self.dropout,
                                   scale=self.scaling)

        attn_output = attn_output.view(bsz, tgt_len, self.num_heads, self.head_dim)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)
        return attn_output, None, past_key_value

    @staticmethod
    def from_native_module(module: nn.Module, process_group: Union[ProcessGroup, List[ProcessGroup]], *args,
                           **kwargs) -> nn.Module:
        r"""
        Convert a native OPT Attention to a Flash Attention.
        """
        # get the attributes
        old_module = module
        module.__class__ = FlashAttentionForOPT
        for key in old_module.__dict__.keys():
            setattr(
                module.__class__,
                key,
                getattr(old_module, key),
            )
        return module
