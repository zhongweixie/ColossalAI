""" PyTorch ChatGLM model. """
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, LayerNorm
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging

from colossalai.pipeline.stage_manager import PipelineStageManager
from tests.kit.model_zoo.transformers.chatglm2_6b.configuration_chatglm import ChatGLMConfig
from tests.kit.model_zoo.transformers.chatglm2_6b.modeling_chatglm import (
    ChatGLMForConditionalGeneration,
    ChatGLMModel,
    GLMBlock,
)


class ChatGLMPipelineForwards:
    '''
    This class serves as a micro library for ChatGLM model forwards under pipeline parallelism.
    '''

    def chatglm_model_forward(
        self: ChatGLMModel,
        input_ids,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        full_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
    ):

        logger = logging.get_logger(__name__)
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # TODO: left the recording kv-value tensors as () or None type, this feature may be added in the future.
        if past_key_values:
            logger.warning_once('Non-empty past_key_values is not supported for pipeline models at the moment.')
            past_key_values = None
        if output_hidden_states:
            logger.warning_once('output_hidden_states=True is not supported for pipeline models at the moment.')
            output_hidden_states = False
        if use_cache:
            logger.warning_once('use_cache=True is not supported for pipeline models at the moment.')
            use_cache = False

        if stage_manager.is_first_stage():
            batch_size, seq_length = input_ids.shape

            if inputs_embeds is None:
                inputs_embeds = self.embedding(input_ids)
            hidden_states = inputs_embeds
        else:
            seq_length, batch_size = hidden_states.shape[:2]

        if self.pre_seq_len is not None:
            if past_key_values is None:
                past_key_values = self.get_prompt(batch_size=batch_size,
                                                  device=input_ids.device,
                                                  dtype=inputs_embeds.dtype)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask.new_ones((batch_size, self.pre_seq_len)), attention_mask],
                                           dim=-1)

        if full_attention_mask is None:
            if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
                full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)
        # Rotary positional embeddings
        rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

        if not past_key_values:
            past_key_values = [None for _ in range(self.num_layers)]

        presents = () if use_cache else None

        if self.encoder.gradient_checkpointing and self.encoder.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                use_cache = False

        all_self_attentions = None
        all_hidden_states = () if output_hidden_states else None
        start_idx, end_idx = stage_index[0], stage_index[1]
        for idx in range(start_idx, end_idx):
            layer = self.encoder._get_layer(idx)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.encoder.gradient_checkpointing and self.encoder.training:
                layer_ret = torch.utils.checkpoint.checkpoint(layer, hidden_states, attention_mask, rotary_pos_emb,
                                                              past_key_values[idx], use_cache)
            else:
                layer_ret = layer(hidden_states,
                                  full_attention_mask,
                                  rotary_pos_emb,
                                  kv_cache=past_key_values[idx],
                                  use_cache=use_cache)

            hidden_states, kv_cache = layer_ret
            if use_cache:
                presents = presents + (kv_cache,)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if stage_manager.is_last_stage():
            # final layer_norm
            if self.encoder.post_layer_norm:
                hidden_states = self.encoder.final_layernorm(hidden_states)
            if not return_dict:
                return tuple(
                    v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=presents,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
            )
        else:
            return {'hidden_states': hidden_states}

    def chatglm_for_conditional_generation_forward(
        self: ChatGLMForConditionalGeneration,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_last_logit: Optional[bool] = False,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
    ):
        logger = logging.get_logger(__name__)

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (return_dict if return_dict is not None else self.config.use_return_dict)

        transformer_outputs = ChatGLMPipelineForwards.chatglm_model_forward(
            self.transformer,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if stage_manager.is_last_stage():
            hidden_states = transformer_outputs[0]
            if return_last_logit:
                hidden_states = hidden_states[-1:]
            lm_logits = self.transformer.output_layer(hidden_states)
            lm_logits = lm_logits.transpose(0, 1).contiguous()

            loss = None
            if labels is not None:
                lm_logits = lm_logits.to(torch.float32)

                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                lm_logits = lm_logits.to(hidden_states.dtype)
                loss = loss.to(hidden_states.dtype)

            if not return_dict:
                output = (lm_logits,) + transformer_outputs[1:]
                return ((loss,) + output) if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=lm_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
            )
        else:
            hidden_states = transformer_outputs.get('hidden_states')
            return {'hidden_states', hidden_states}
