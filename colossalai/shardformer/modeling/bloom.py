import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.models.bloom.modeling_bloom import (
    BloomAttention,
    BloomBlock,
    BloomForCausalLM,
    BloomForQuestionAnswering,
    BloomForSequenceClassification,
    BloomForTokenClassification,
    BloomModel,
)
from transformers.utils import logging

from colossalai.kernel.triton.context_attention import bloom_context_attn_fwd
from colossalai.kernel.triton.copy_kv_cache_dest import copy_kv_cache_to_dest
from colossalai.kernel.triton.token_attention_kernel import token_attention_fwd
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer.inference import BatchInferState


def build_bloom_alibi_tensor_fn(process_group: ProcessGroup) -> torch.Tensor:

    def build_bloom_alibi_tensor(self, attention_mask: torch.Tensor, num_heads: int,
                                 dtype: torch.dtype) -> torch.Tensor:
        """
        Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
        relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
        `softmax(l+a) = softmax(l)`. Based on
        https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
        TODO @thomasw21 this doesn't work as nicely due to the masking strategy, and so masking varies slightly.

        Args:
        Returns tensor shaped (batch_size * num_heads, 1, max_seq_len)
            attention_mask (`torch.Tensor`):
                Token-wise attention mask, this should be of shape (batch_size, max_seq_len).
            num_heads (`int`, *required*):
                number of heads
            dtype (`torch.dtype`, *optional*, default=`torch.bfloat16`):
                dtype of the output tensor
        """
        import math

        if dist.is_initialized():
            world_size = dist.get_world_size(process_group)
            num_heads = num_heads * world_size

        batch_size, seq_length = attention_mask.shape
        closest_power_of_2 = 2**math.floor(math.log2(num_heads))
        base = torch.tensor(2**(-(2**-(math.log2(closest_power_of_2) - 3))),
                            device=attention_mask.device,
                            dtype=torch.float32)
        powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
        slopes = torch.pow(base, powers)

        if closest_power_of_2 != num_heads:
            extra_base = torch.tensor(2**(-(2**-(math.log2(2 * closest_power_of_2) - 3))),
                                      device=attention_mask.device,
                                      dtype=torch.float32)
            num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
            extra_powers = torch.arange(1,
                                        1 + 2 * num_remaining_heads,
                                        2,
                                        device=attention_mask.device,
                                        dtype=torch.int32)
            slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

        # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
        # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
        # => here we set (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
        # => the query_length dimension will then be broadcasted correctly
        # This is more or less identical to T5's relative position bias:
        # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
        arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
        alibi = slopes[..., None] * arange_tensor
        if dist.is_initialized():
            num_heads_per_rank = int(num_heads / dist.get_world_size(process_group))
            offset = dist.get_rank(process_group) * num_heads_per_rank
            alibi = alibi.view(batch_size, num_heads, 1, seq_length)
            alibi = alibi[:, offset:num_heads_per_rank + offset, :, :]
            return alibi.reshape(batch_size * num_heads_per_rank, 1, seq_length).to(dtype)
        else:
            return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)

    return build_bloom_alibi_tensor


def generate_alibi(n_head, dtype=torch.float16):
    """
    This method is originally the `build_alibi_tensor` function
    in `transformers/models/bloom/modeling_bloom.py`
    of the huggingface/transformers GitHub repository.

    Copyright 2023 ModelTC Team
    Copyright 2022 HuggingFace Inc. team and BigScience workshop

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    """

    def get_slopes(n):

        def get_slopes_power_of_2(n):
            start = 2**(-(2**-(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n))
            return (get_slopes_power_of_2(closest_power_of_2) +
                    get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2])

    slopes = torch.Tensor(get_slopes(n_head))
    head_alibi = slopes.to(dtype)
    return head_alibi    # 1 * num_heads


def generate_alibi_2(n_head, dtype=torch.float16):

    def get_slopes_power_of_2(n):
        start = 2**(-(2**-(math.log2(n) - 3)))
        return [start * start**i for i in range(n)]

    def get_slopes(n):
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n))
            slopes_power_of_2 = get_slopes_power_of_2(closest_power_of_2)
            slopes_double = get_slopes(2 * closest_power_of_2)
            slopes_combined = slopes_power_of_2 + slopes_double[0::2][:n - closest_power_of_2]
            return slopes_combined

    slopes = torch.tensor(get_slopes(n_head), dtype=dtype)
    return slopes


class BloomPipelineForwards:
    '''
    This class serves as a micro library for bloom pipeline forwards.
    '''

    @staticmethod
    def bloom_model_forward(
        self: BloomModel,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor, ...], 'BaseModelOutputWithPastAndCrossAttentions']:

        logger = logging.get_logger(__name__)

        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # add warnings here
        if output_attentions:
            logger.warning_once('output_attentions=True is not supported for pipeline models at the moment.')
            output_attentions = False
        if output_hidden_states:
            logger.warning_once('output_hidden_states=True is not supported for pipeline models at the moment.')
            output_hidden_states = False
        if use_cache:
            logger.warning_once('use_cache=True is not supported for pipeline models at the moment.')
            use_cache = False
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N

        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        # case: First stage of training
        if stage_manager.is_first_stage():
            # check input_ids and inputs_embeds
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            elif input_ids is not None:
                batch_size, seq_length = input_ids.shape
            elif inputs_embeds is not None:
                batch_size, seq_length, _ = inputs_embeds.shape
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")

            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)

            hidden_states = self.word_embeddings_layernorm(inputs_embeds)
            # initialize in the first stage and then pass to the next stage
        else:
            input_shape = hidden_states.shape[:-1]
            batch_size, seq_length = input_shape

        # extra recording tensor should be generated in the first stage

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                use_cache = False

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))
        # Compute alibi tensor: check build_alibi_tensor documentation,build for every stage
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[2]    # source_len

            seq_length_with_past = seq_length_with_past + past_key_values_length
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(hidden_states.device)

        alibi = self.build_alibi_tensor(attention_mask, self.num_heads, dtype=hidden_states.dtype)

        # causal_mask is constructed every stage and its input is passed through different stages
        causal_mask = self._prepare_attn_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            past_key_values_length=past_key_values_length,
        )

        start_idx, end_idx = stage_index[0], stage_index[1]
        for i, (block, layer_past) in enumerate(zip(self.h[start_idx:end_idx], past_key_values[start_idx:end_idx]),
                                                start=start_idx):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache=use_cache, output_attentions=output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    alibi,
                    causal_mask,
                    layer_past,
                    head_mask[i],
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=causal_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi=alibi,
                )

            hidden_states = outputs[0]

            if use_cache is True:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + \
                    (outputs[2 if use_cache else 1],)

        if stage_manager.is_last_stage():
            # Add last hidden state
            hidden_states = self.ln_f(hidden_states)

        # TODO(jianghai): deal with all_hidden_states, all_self_attentions, presents
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if stage_manager.is_last_stage():
            if not return_dict:
                return tuple(
                    v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

            # attention_mask is not returned ; presents = past_key_values
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=presents,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
            )
        else:
            # always return dict for imediate stage
            return {'hidden_states': hidden_states}

    @staticmethod
    def bloom_for_causal_lm_forward(self: BloomForCausalLM,
                                    input_ids: Optional[torch.LongTensor] = None,
                                    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
                                    attention_mask: Optional[torch.Tensor] = None,
                                    head_mask: Optional[torch.Tensor] = None,
                                    inputs_embeds: Optional[torch.Tensor] = None,
                                    labels: Optional[torch.Tensor] = None,
                                    use_cache: Optional[bool] = None,
                                    output_attentions: Optional[bool] = None,
                                    output_hidden_states: Optional[bool] = None,
                                    return_dict: Optional[bool] = None,
                                    stage_manager: Optional[PipelineStageManager] = None,
                                    hidden_states: Optional[torch.FloatTensor] = None,
                                    stage_index: Optional[List[int]] = None,
                                    **deprecated_arguments):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        logger = logging.get_logger(__name__)

        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # TODO(jianghai): left the recording kv-value tensors as () or None type, this feature may be added in the future.
        if output_attentions:
            logger.warning_once('output_attentions=True is not supported for pipeline models at the moment.')
            output_attentions = False
        if output_hidden_states:
            logger.warning_once('output_hidden_states=True is not supported for pipeline models at the moment.')
            output_hidden_states = False

        transformer_outputs = BloomPipelineForwards.bloom_model_forward(self.transformer,
                                                                        input_ids,
                                                                        past_key_values=past_key_values,
                                                                        attention_mask=attention_mask,
                                                                        head_mask=head_mask,
                                                                        inputs_embeds=inputs_embeds,
                                                                        use_cache=use_cache,
                                                                        output_attentions=output_attentions,
                                                                        output_hidden_states=output_hidden_states,
                                                                        return_dict=return_dict,
                                                                        stage_manager=stage_manager,
                                                                        hidden_states=hidden_states,
                                                                        stage_index=stage_index)
        past_key_values = None
        all_hidden_states = None
        all_self_attentions = None
        all_cross_attentions = None
        if stage_manager.is_last_stage():
            hidden_states = transformer_outputs[0]
            lm_logits = self.lm_head(hidden_states)

            loss = None
            if labels is not None:
                # move labels to correct device to enable model parallelism
                labels = labels.to(lm_logits.device)
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                batch_size, seq_length, vocab_size = shift_logits.shape
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(batch_size * seq_length, vocab_size),
                                shift_labels.view(batch_size * seq_length))

            if not return_dict:
                output = (lm_logits,) + transformer_outputs[1:]
                return ((loss,) + output) if loss is not None else output

            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=lm_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
            )
        else:
            hidden_states = transformer_outputs.get('hidden_states')
            return {'hidden_states': hidden_states}

    @staticmethod
    def bloom_for_sequence_classification_forward(
        self: BloomForSequenceClassification,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
        **deprecated_arguments,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        logger = logging.get_logger(__name__)

        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # TODO(jianghai): left the recording kv-value tensors as () or None type, this feature may be added in the future.
        if output_attentions:
            logger.warning_once('output_attentions=True is not supported for pipeline models at the moment.')
            output_attentions = False
        if output_hidden_states:
            logger.warning_once('output_hidden_states=True is not supported for pipeline models at the moment.')
            output_hidden_states = False

        transformer_outputs = BloomPipelineForwards.bloom_model_forward(
            self.transformer,
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            stage_manager=stage_manager,
            hidden_states=hidden_states,
            stage_index=stage_index,
        )
        past_key_values = None
        all_hidden_states = None
        all_self_attentions = None
        all_cross_attentions = None
        if stage_manager.is_last_stage():
            batch_size = hidden_states.shape[0]
            # update batch size
            hidden_states = transformer_outputs[0]
            logits = self.score(hidden_states)

            if self.config.pad_token_id is None and batch_size != 1:
                raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
            if self.config.pad_token_id is None:
                sequence_lengths = -1
            else:
                if input_ids is not None:
                    sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
                else:
                    sequence_lengths = -1
                    logger.warning(
                        f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                        "unexpected if using padding tokens in conjunction with `inputs_embeds.`")

            pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

            loss = None
            if labels is not None:
                if self.config.problem_type is None:
                    if self.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    loss_fct = MSELoss()
                    if self.num_labels == 1:
                        loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                    else:
                        loss = loss_fct(pooled_logits, labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(pooled_logits, labels)
                elif self.config.problem_type == "multi_label_classification":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(pooled_logits, labels)
            if not return_dict:
                output = (pooled_logits,) + transformer_outputs[1:]
                return ((loss,) + output) if loss is not None else output

            return SequenceClassifierOutputWithPast(
                loss=loss,
                logits=pooled_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
            )
        else:
            hidden_states = transformer_outputs.get('hidden_states')
            return {'hidden_states': hidden_states}

    @staticmethod
    def bloom_for_token_classification_forward(
        self: BloomForTokenClassification,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
        **deprecated_arguments,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        logger = logging.get_logger(__name__)

        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # TODO(jianghai): left the recording kv-value tensors as () or None type, this feature may be added in the future.
        if output_attentions:
            logger.warning_once('output_attentions=True is not supported for pipeline models at the moment.')
            output_attentions = False
        if output_hidden_states:
            logger.warning_once('output_hidden_states=True is not supported for pipeline models at the moment.')
            output_hidden_states = False

        transformer_outputs = BloomPipelineForwards.bloom_model_forward(
            self.transformer,
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            stage_manager=stage_manager,
            hidden_states=hidden_states,
            stage_index=stage_index,
        )
        past_key_values = None
        all_hidden_states = None
        all_self_attentions = None
        all_cross_attentions = None

        if stage_manager.is_last_stage():
            hidden_states = transformer_outputs[0]
            hidden_states = self.dropout(hidden_states)
            logits = self.classifier(hidden_states)

            loss = None
            if labels is not None:
                # move labels to correct device to enable model parallelism
                labels = labels.to(logits.device)
                batch_size, seq_length = labels.shape
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(batch_size * seq_length, self.num_labels),
                                labels.view(batch_size * seq_length))

            if not return_dict:
                output = (logits,) + transformer_outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return TokenClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
            )
        else:
            hidden_states = transformer_outputs.get('hidden_states')
            return {'hidden_states': hidden_states}

    @staticmethod
    def bloom_for_question_answering_forward(
        self: BloomForQuestionAnswering,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
    ):
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        logger = logging.get_logger(__name__)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # TODO(jianghai): left the recording kv-value tensors as () or None type, this feature may be added in the future.
        if output_attentions:
            logger.warning_once('output_attentions=True is not supported for pipeline models at the moment.')
            output_attentions = False
        if output_hidden_states:
            logger.warning_once('output_hidden_states=True is not supported for pipeline models at the moment.')
            output_hidden_states = False

        outputs = BloomPipelineForwards.bloom_model_forward(
            self.transformer,
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            stage_manager=stage_manager,
            hidden_states=hidden_states,
            stage_index=stage_index,
        )
        past_key_values = None
        all_hidden_states = None
        all_self_attentions = None
        all_cross_attentions = None

        if stage_manager.is_last_stage():
            sequence_output = outputs[0]
            logits = self.qa_outputs(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1).contiguous()
            end_logits = end_logits.squeeze(-1).contiguous()

            total_loss = None
            if start_positions is not None and end_positions is not None:
                # If we are on multi-GPU, split add a dimension
                if len(start_positions.size()) > 1:
                    start_positions = start_positions.squeeze(-1)
                if len(end_positions.size()) > 1:
                    end_positions = end_positions.squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.size(1)
                start_positions = start_positions.clamp(0, ignored_index)
                end_positions = end_positions.clamp(0, ignored_index)

                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2

            if not return_dict:
                output = (start_logits, end_logits) + outputs[2:]
                return ((total_loss,) + output) if total_loss is not None else output

            return QuestionAnsweringModelOutput(
                loss=total_loss,
                start_logits=start_logits,
                end_logits=end_logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            hidden_states = outputs.get('hidden_states')
            return {'hidden_states': hidden_states}


class BloomInferenceForwards:
    """
    This class serves a micro library for bloom inference forwards
    """

    @staticmethod
    def bloom_model_forward(
        self: BloomModel,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        infer_state: Optional[BatchInferState] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:

        logger = logging.get_logger(__name__)

        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # # initialize BatchInferState to track necessary states during current model forward
        # infer_state = BatchInferState()
        # infer_state.batch_size = batch_size
        # # TODO: dummy implementation here for testing, assume all inputs same length
        # infer_state.total_token_num = batch_size * seq_length
        # infer_state.block_loc = self.block_loc
        # infer_state.start_loc = self.b_start_loc
        # infer_state.seq_len = self.b_seq_len

        # still need to keep past_key_values to fit original forward flow·
        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                use_cache = False

        # NOTE determine if BatchInferState is passed in via arg
        #      if not, get the attr binded to the model
        # We might wantto remove setattr later
        if infer_state is None:
            infer_state = self.infer_state

        # Compute alibi tensor: check build_alibi_tensor documentation
        seq_length_with_past = seq_length
        past_key_values_length = 0
        # if self.cache_manager.past_key_values_length > 0:
        if infer_state.cache_manager.past_key_values_length > 0:
            # TODO dummy but work, revise it
            past_key_values_length = infer_state.cache_manager.past_key_values_length
            # past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        # infer_state.cache_manager = self.cache_manager

        if use_cache and seq_length != 1:
            # NOTE assuem prefill stage
            # allocate memory block
            infer_state.is_context_stage = True    # set prefill stage, notify attention layer
            infer_state.context_mem_index = infer_state.cache_manager.alloc(infer_state.total_token_num)
            BatchInferState.init_block_loc(infer_state.block_loc, infer_state.seq_len, seq_length,
                                           infer_state.context_mem_index)
        else:
            # TODO handle the condition that no contiguous memory presents
            alloc_mem = infer_state.cache_manager.alloc_contiguous(batch_size)
            if alloc_mem is not None:
                infer_state.decode_is_contiguous = True
                infer_state.decode_mem_index = alloc_mem[0]
                infer_state.decode_mem_start = alloc_mem[1]
                infer_state.decode_mem_end = alloc_mem[2]
                infer_state.block_loc[:, seq_length_with_past - 1] = infer_state.decode_mem_index
            else:
                print(f" *** Encountered allocation non-contiguous")
                print(
                    f"    infer_state.cache_manager.past_key_values_length: {infer_state.cache_manager.past_key_values_length}"
                )
                infer_state.decode_is_contiguous = False
                alloc_mem = infer_state.cache_manager.alloc(batch_size)
                infer_state.decode_mem_index = alloc_mem
                # infer_state.decode_key_buffer = torch.empty((batch_size, self.tp_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
                # infer_state.decode_value_buffer = torch.empty((batch_size, self.tp_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
                infer_state.block_loc[:, seq_length_with_past - 1] = infer_state.decode_mem_index

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(hidden_states.device)

        # NOTE we might want to store a single 1D alibi(length is #heads) in model
        alibi = generate_alibi(self.num_heads).contiguous().cuda()
        # alibi = self.build_alibi_tensor(attention_mask, self.num_heads, dtype=hidden_states.dtype)

        causal_mask = self._prepare_attn_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            past_key_values_length=past_key_values_length,
        )

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                # FIXME: currently our KV cache manager does not handle this condition
                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache=use_cache, output_attentions=output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    alibi,
                    causal_mask,
                    layer_past,
                    head_mask[i],
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=causal_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi=alibi,
                    infer_state=infer_state,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # NOTE: here we still to update indices of kv cache block
        # TODO: remove this part, instead, better to pass the BatchInferState from model forward,
        #       and update these information in engine.generate after model foward called
        infer_state.start_loc = infer_state.start_loc + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
        infer_state.seq_len += 1
        infer_state.decode_layer_id = 0

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,    # should always be (None, None, ..., None)
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    @staticmethod
    def bloom_for_causal_lm_forward(self: BloomForCausalLM,
                                    input_ids: Optional[torch.LongTensor] = None,
                                    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
                                    attention_mask: Optional[torch.Tensor] = None,
                                    head_mask: Optional[torch.Tensor] = None,
                                    inputs_embeds: Optional[torch.Tensor] = None,
                                    labels: Optional[torch.Tensor] = None,
                                    use_cache: Optional[bool] = None,
                                    output_attentions: Optional[bool] = None,
                                    output_hidden_states: Optional[bool] = None,
                                    return_dict: Optional[bool] = None,
                                    infer_state: Optional[BatchInferState] = None,
                                    **deprecated_arguments):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        logger = logging.get_logger(__name__)

        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = BloomInferenceForwards.bloom_model_forward(self.transformer,
                                                                         input_ids,
                                                                         past_key_values=past_key_values,
                                                                         attention_mask=attention_mask,
                                                                         head_mask=head_mask,
                                                                         inputs_embeds=inputs_embeds,
                                                                         use_cache=use_cache,
                                                                         output_attentions=output_attentions,
                                                                         output_hidden_states=output_hidden_states,
                                                                         return_dict=return_dict,
                                                                         infer_state=infer_state)
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(batch_size * seq_length, vocab_size),
                            shift_labels.view(batch_size * seq_length))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def bloom_for_causal_lm_prepare_inputs_for_generation(
        self: BloomForCausalLM,
        input_ids: torch.LongTensor,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        # only last token for input_ids if past is not None
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)

            # NOTE we won't use past key values here
            # the cache may be in the stardard format (e.g. in contrastive search), convert to bloom's format if needed
            # if past_key_values[0][0].shape[0] == input_ids.shape[0]:
            #     past_key_values = self._convert_to_bloom_cache(past_key_values)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        })
        return model_inputs

    # replace decoder layer forward:
    #   used to replace BloomBlock.forward
    @staticmethod
    def bloom_block_forward(
        self: BloomBlock,
        hidden_states: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        infer_state: Optional[BatchInferState] = None,
    ):
        # hidden_states: [batch_size, seq_length, hidden_size]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Layer norm post the self attention.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # Self attention.
        attn_outputs = self.self_attention(
            layernorm_output,
            residual,
            layer_past=layer_past,
            attention_mask=attention_mask,
            alibi=alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            infer_state=infer_state,
        )

        attention_output = attn_outputs[0]

        outputs = attn_outputs[1:]

        layernorm_output = self.post_attention_layernorm(attention_output)

        # Get residual
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = attention_output

        # MLP.
        output = self.mlp(layernorm_output, residual)

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs    # hidden_states, present, attentions

    # replace attention forward:
    #   used to replace BloomAttention.forward
    @staticmethod
    def bloom_attention_forward(
        self: BloomAttention,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        infer_state: Optional[BatchInferState] = None,
    ):

        fused_qkv = self.query_key_value(hidden_states)    # [batch_size, seq_length, 3 x hidden_size]

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)
        batch_size, q_length, H, D_HEAD = query_layer.shape
        k = key_layer.reshape(-1, H, D_HEAD)    # batch_size * q_length, H, D_HEAD, q_lenth == 1
        v = value_layer.reshape(-1, H, D_HEAD)    # batch_size * q_length, H, D_HEAD, q_lenth == 1

        mem_manager = infer_state.cache_manager
        layer_id = infer_state.decode_layer_id

        if infer_state.is_context_stage:
            # context process
            max_input_len = q_length
            b_start_loc = infer_state.start_loc
            b_seq_len = infer_state.seq_len[:batch_size]
            q = query_layer.reshape(-1, H, D_HEAD)

            copy_kv_cache_to_dest(k, infer_state.context_mem_index, mem_manager.key_buffer[layer_id])
            copy_kv_cache_to_dest(v, infer_state.context_mem_index, mem_manager.value_buffer[layer_id])

            # output = self.output[:batch_size*q_length, :, :]
            output = torch.empty_like(q)

            # temp for testing
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"  cp3")
                print(f"  q.shape: {q.shape}")
                print(f"  k.shape: {k.shape}")
                print(f"  v.shape: {v.shape}")
                print(f"  output.shape: {output.shape}")
                print(f"  b_start_loc: {b_start_loc}")
                print(f"  b_seq_len: {b_seq_len}")
                print(f"  max_input_len: {max_input_len}")
                print(f"  alibi: {alibi}")

            bloom_context_attn_fwd(q, k, v, output, b_start_loc, b_seq_len, max_input_len, alibi)

            context_layer = output.view(batch_size, q_length, H * D_HEAD)
            # FIXME might want to revise
            #   need some way to record the length of past key values cache
            #   since we won't return past_key_value_cache right now
            if layer_id == 0:    # once per model.forward
                infer_state.cache_manager.past_key_values_length = q_length    # seq_len
        else:
            # query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
            # need shape: batch_size, H, D_HEAD (q_length == 1), input q shape : (batch_size, q_length(1), H, D_HEAD)
            assert q_length == 1, "for non-context process, we only support q_length == 1"
            q = query_layer.reshape(-1, H, D_HEAD)

            if infer_state.decode_is_contiguous:
                # if decode is contiguous, then we copy to key cache and value cache in cache manager directly
                cache_k = infer_state.cache_manager.key_buffer[layer_id][
                    infer_state.decode_mem_start:infer_state.decode_mem_end, :, :]
                cache_v = infer_state.cache_manager.value_buffer[layer_id][
                    infer_state.decode_mem_start:infer_state.decode_mem_end, :, :]
                cache_k.copy_(k)
                cache_v.copy_(v)
            else:
                # if decode is not contiguous, use triton kernel to copy key and value cache
                # k, v shape: [batch_size, num_heads, head_dim/embed_size_per_head]
                # TODO clean comments
                # destindex_copy_kv(k, infer_state.decode_mem_index, mem_manager.key_buffer[layer_id])
                # destindex_copy_kv(v, infer_state.decode_mem_index, mem_manager.value_buffer[layer_id])
                copy_kv_cache_to_dest(k, infer_state.decode_mem_index, mem_manager.key_buffer[layer_id])
                copy_kv_cache_to_dest(v, infer_state.decode_mem_index, mem_manager.value_buffer[layer_id])

            b_start_loc = infer_state.start_loc[:batch_size]
            b_loc = infer_state.block_loc[:batch_size, :]
            b_seq_len = infer_state.seq_len[:batch_size]
            max_len_in_batch = mem_manager.past_key_values_length + q_length
            output = torch.empty_like(q)
            token_attention_fwd(q, mem_manager.key_buffer[layer_id], mem_manager.value_buffer[layer_id], output, b_loc,
                                b_start_loc, b_seq_len, max_len_in_batch, alibi)

            context_layer = output.view(batch_size, q_length, H * D_HEAD)
            # FIXME might want to revise (same as above one)
            #   need some way to record the length of past key values cache
            #   since we won't return past_key_value_cache right now
            if layer_id == 0:    # once per model.forward
                assert infer_state.cache_manager.past_key_values_length != 0
                infer_state.cache_manager.past_key_values_length += q_length    # += 1

        # update layer id
        infer_state.decode_layer_id += 1

        # NOTE: always set present as none for now, instead of returning past key value to the next decoding,
        #       we create the past key value pair from the cache manager
        present = None

        # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
        if self.pretraining_tp > 1 and self.slow_but_exact:
            slices = self.hidden_size / self.pretraining_tp
            output_tensor = torch.zeros_like(context_layer)
            for i in range(self.pretraining_tp):
                output_tensor = output_tensor + F.linear(
                    context_layer[:, :, int(i * slices):int((i + 1) * slices)],
                    self.dense.weight[:, int(i * slices):int((i + 1) * slices)],
                )
        else:
            output_tensor = self.dense(context_layer)

        # dropout is not required here during inference
        output_tensor = residual + output_tensor

        outputs = (output_tensor, present)
        assert output_attentions is False, "we do not support output_attentions at this time"

        return outputs


def get_bloom_flash_attention_forward(enabel_jit_fused=False):

    try:
        from xformers.ops import memory_efficient_attention as me_attention
    except:
        raise ImportError("Error: xformers module is not installed. Please install it to use flash attention.")
    from transformers.models.bloom.modeling_bloom import BloomAttention

    def forward(
        self: BloomAttention,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):

        fused_qkv = self.query_key_value(hidden_states)
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)
        batch_size, tgt_len, _ = hidden_states.size()
        assert tgt_len % 4 == 0, "Flash Attention Error: The sequence length should be a multiple of 4."

        _, kv_length, _, _ = key_layer.size()

        proj_shape = (batch_size, tgt_len, self.num_heads, self.head_dim)
        query_layer = query_layer.contiguous().view(*proj_shape)
        key_layer = key_layer.contiguous().view(*proj_shape)
        value_layer = value_layer.contiguous().view(*proj_shape)

        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, head_dim, kv_length]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            key_layer = torch.cat((past_key, key_layer), dim=1)
            value_layer = torch.cat((past_value, value_layer), dim=1)

        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None

        tgt_len = key_layer.size()[1]

        attention_numerical_mask = torch.zeros((batch_size, self.num_heads, tgt_len, kv_length),
                                               dtype=torch.float32,
                                               device=query_layer.device,
                                               requires_grad=True)
        attention_numerical_mask = attention_numerical_mask + alibi.view(batch_size, self.num_heads, 1,
                                                                         kv_length) * self.beta
        attention_numerical_mask = torch.masked_fill(attention_numerical_mask, attention_mask,
                                                     torch.finfo(torch.float32).min)

        context_layer = me_attention(query_layer,
                                     key_layer,
                                     value_layer,
                                     attn_bias=attention_numerical_mask,
                                     scale=self.inv_norm_factor,
                                     p=self.attention_dropout.p)
        context_layer = context_layer.reshape(-1, kv_length, self.hidden_size)
        if self.pretraining_tp > 1 and self.slow_but_exact:
            slices = self.hidden_size / self.pretraining_tp
            output_tensor = torch.zeros_like(context_layer)
            for i in range(self.pretraining_tp):
                output_tensor = output_tensor + F.linear(
                    context_layer[:, :, int(i * slices):int((i + 1) * slices)],
                    self.dense.weight[:, int(i * slices):int((i + 1) * slices)],
                )
        else:
            output_tensor = self.dense(context_layer)

        # TODO to replace with the bias_dropout_add function in jit
        output_tensor = self.dropout_add(output_tensor, residual, self.hidden_dropout, self.training)
        outputs = (output_tensor, present, None)

        return outputs

    return forward


def get_jit_fused_bloom_attention_forward():

    from transformers.models.bloom.modeling_bloom import BloomAttention

    def forward(
        self: BloomAttention,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        fused_qkv = self.query_key_value(hidden_states)    # [batch_size, seq_length, 3 x hidden_size]

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, q_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        key_layer = key_layer.permute(0, 2, 3, 1).reshape(batch_size * self.num_heads, self.head_dim, q_length)
        value_layer = value_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, head_dim, kv_length]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            key_layer = torch.cat((past_key, key_layer), dim=2)
            value_layer = torch.cat((past_value, value_layer), dim=1)

        _, _, kv_length = key_layer.shape

        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None

        # [batch_size * num_heads, q_length, kv_length]
        # we use `torch.Tensor.baddbmm` instead of `torch.baddbmm` as the latter isn't supported by TorchScript v1.11
        matmul_result = alibi.baddbmm(
            batch1=query_layer,
            batch2=key_layer,
            beta=self.beta,
            alpha=self.inv_norm_factor,
        )

        # change view to [batch_size, num_heads, q_length, kv_length]
        attention_scores = matmul_result.view(batch_size, self.num_heads, q_length, kv_length)

        # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
        input_dtype = attention_scores.dtype
        # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
        if input_dtype == torch.float16:
            attention_scores = attention_scores.to(torch.float)
        attn_weights = torch.masked_fill(attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min)
        attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(input_dtype)

        # [batch_size, num_heads, q_length, kv_length]
        attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # change view [batch_size x num_heads, q_length, kv_length]
        attention_probs_reshaped = attention_probs.view(batch_size * self.num_heads, q_length, kv_length)

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = torch.bmm(attention_probs_reshaped, value_layer)

        # change view [batch_size, num_heads, q_length, head_dim]
        context_layer = self._merge_heads(context_layer)

        # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
        if self.pretraining_tp > 1 and self.slow_but_exact:
            slices = self.hidden_size / self.pretraining_tp
            output_tensor = torch.zeros_like(context_layer)
            for i in range(self.pretraining_tp):
                output_tensor = output_tensor + F.linear(
                    context_layer[:, :, int(i * slices):int((i + 1) * slices)],
                    self.dense.weight[:, int(i * slices):int((i + 1) * slices)],
                )
        else:
            output_tensor = self.dense(context_layer)

        output_tensor = self.dropout_add(output_tensor, residual, self.hidden_dropout, self.training)

        outputs = (output_tensor, present)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs

    return forward


def get_jit_fused_bloom_mlp_forward():

    from transformers.models.bloom.modeling_bloom import BloomMLP

    def forward(self: BloomMLP, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        hidden_states = self.gelu_impl(self.dense_h_to_4h(hidden_states))

        if self.pretraining_tp > 1 and self.slow_but_exact:
            intermediate_output = torch.zeros_like(residual)
            slices = self.dense_4h_to_h.weight.shape[-1] / self.pretraining_tp
            for i in range(self.pretraining_tp):
                intermediate_output = intermediate_output + F.linear(
                    hidden_states[:, :, int(i * slices):int((i + 1) * slices)],
                    self.dense_4h_to_h.weight[:, int(i * slices):int((i + 1) * slices)],
                )
        else:
            intermediate_output = self.dense_4h_to_h(hidden_states)
        output = self.dropout_add(intermediate_output, residual, self.hidden_dropout, self.training)
        return output

    return forward


def get_jit_fused_bloom_gelu_forward():

    from transformers.models.bloom.modeling_bloom import BloomGelu

    from colossalai.kernel.jit.bias_gelu import GeLUFunction as JitGeLUFunction

    def forward(self: BloomGelu, x: torch.Tensor) -> torch.Tensor:
        bias = torch.zeros_like(x)
        if self.training:
            return JitGeLUFunction.apply(x, bias)
        else:
            return self.bloom_gelu_forward(x, bias)

    return forward
