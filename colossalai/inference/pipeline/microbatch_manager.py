from enum import Enum
from typing import Dict, Tuple

import torch

__all__ = 'MicroBatchManager'


class Status(Enum):
    PREFILL = 1
    GENERATE = 2
    DONE = 3


class MicroBatchDescription():

    def __init__(
        self,
        inputs_dict: Dict[str, torch.Tensor],
        output_dict: Dict[str, torch.Tensor],
        new_length: int,
    ) -> None:
        assert output_dict.get('hidden_states') is not None
        self.mb_length = output_dict['hidden_states'].shape[-2]
        self.target_length = self.mb_length + new_length
        self.kv_cache = ()

    def update(self, output_dict: Dict[str, torch.Tensor] = None, new_token: torch.Tensor = None):
        if output_dict is not None:
            self._update_kvcache(output_dict['past_key_values'])

    def _update_kvcache(self, kv_cache: Tuple):
        assert type(kv_cache) == tuple
        self.kv_cache = kv_cache

    @property
    def state(self):
        """
        Return the state of current micro batch, when current length is equal to target length,
        the state is DONE, otherwise GENERATE

        """
        if self.cur_length == self.target_length:
            return Status.DONE
        else:
            return Status.GENERATE

    @property
    def cur_length(self):
        """
        Return the current sequnence length of micro batch

        """
        pass


class HeadMicroBatchDescription(MicroBatchDescription):

    def __init__(self, inputs_dict: Dict[str, torch.Tensor], output_dict: Dict[str, torch.Tensor],
                 new_length: int) -> None:
        super().__init__(inputs_dict, output_dict, new_length)
        assert inputs_dict is not None
        assert inputs_dict.get('input_ids') is not None and inputs_dict.get('attention_mask') is not None
        self.input_ids = inputs_dict['input_ids']
        self.attn_mask = inputs_dict['attention_mask']
        self.new_tokens = None

    def update(self, output_dict: Dict[str, torch.Tensor] = None, new_token: torch.Tensor = None):
        super().update(output_dict, new_token)
        if new_token is not None:
            self._update_newtokens(new_token)
        if self.state is not Status.DONE and new_token is not None:
            self._update_attnmask()

    def _update_newtokens(self, new_token: torch.Tensor):
        if self.new_tokens is None:
            self.new_tokens = new_token
        else:
            self.new_tokens = torch.cat([self.new_tokens, new_token], dim=-1)

    def _update_attnmask(self):
        self.attn_mask = torch.cat(
            (self.attn_mask, torch.ones((self.attn_mask.shape[0], 1), dtype=torch.int64, device='cuda')), dim=-1)

    @property
    def cur_length(self):
        """
        When there is no new_token, the length is mb_length, otherwise the sequence length is `mb_length` plus the length of new_token

        """
        if self.new_tokens is None:
            return self.mb_length
        else:
            return self.mb_length + len(self.new_tokens[0])


class BodyMicroBatchDescription(MicroBatchDescription):

    def __init__(self, inputs_dict: Dict[str, torch.Tensor], output_dict: Dict[str, torch.Tensor],
                 new_length: int) -> None:
        super().__init__(inputs_dict, output_dict, new_length)

    def update(self, output_dict: Dict[str, torch.Tensor] = None, new_token: torch.Tensor = None):
        super().update(output_dict, new_token)

    @property
    def cur_length(self):
        """
        When there is no kv_cache, the length is mb_length, otherwise the sequence length is `kv_cache[0][0].shape[-2]` plus 1

        """
        if len(self.kv_cache) == 0:
            return self.mb_length
        else:
            return self.kv_cache[0][0].shape[-2] + 1


class MicroBatchManager():
    '''
    MicroBatchManager is a class that manages the micro batch.

    Args:
        stage (int): stage id of current stage.
        new_length (int): the new length of the input sequence.
        micro_batch_size (int): the micro batch size.
        micro_batch_buffer_size (int): the buffer size for micro batch. Normally, it should be the same as the number of pipeline stages.

    '''

    def __init__(self, stage: int, new_length: int, micro_batch_size: int, micro_batch_buffer_size: int):
        self.stage = stage
        self.new_length = new_length
        self.micro_batch_size = micro_batch_size
        self.buffer_size = micro_batch_buffer_size
        self.mb_descrption_buffer = {}
        self.new_tokens_buffer = {}
        self.idx = 0

    def step(self, inputs_dict=None, output_dict: Dict[str, torch.Tensor] = None, new_token: torch.Tensor = None):
        """
        Update the state if microbatch manager, 2 conditions.
        1. For first stage in PREFILL, receive inputs and outputs, `_add_descrption` will save its inputs.
        2. For other conditon, only receive the output of previous stage, and update the descrption.

        Args:
            inputs_dict (Dict[str, torch.Tensor]): the inputs of current stage. The key should have `input_ids` and `attention_mask`.
            output_dict (Dict[str, torch.Tensor]): the outputs of previous stage. The key should have `hidden_states` and `past_key_values`.
            new_token (torch.Tensor): the new token generated by current stage.
        """
        # Add descrption first if the descrption is None
        if inputs_dict is None and output_dict is None and new_token is None:
            return Status.PREFILL
        if self.mb_descrption_buffer.get(self.idx) is None:
            self._add_descrption(inputs_dict, output_dict)
        self.cur_descrption.update(output_dict, new_token)
        return self.cur_state

    def export_new_tokens(self):
        list = [i.new_tokens.tolist()[0] for i in self.mb_descrption_buffer.values()]
        return list

    def is_micro_batch_done(self):
        if len(self.mb_descrption_buffer) == 0:
            return False
        for mb in self.mb_descrption_buffer.values():
            if mb.state != Status.DONE:
                return False
        return True

    def clear(self):
        self.mb_descrption_buffer.clear()

    def next(self):
        self.idx = (self.idx + 1) % self.buffer_size

    def _add_descrption(self, inputs_dict: Dict[str, torch.Tensor], output_dict: Dict[str, torch.Tensor]):
        if self.stage == 0:
            self.mb_descrption_buffer[self.idx] = HeadMicroBatchDescription(inputs_dict, output_dict, self.new_length)
        else:
            self.mb_descrption_buffer[self.idx] = BodyMicroBatchDescription(inputs_dict, output_dict, self.new_length)

    def _remove_descrption(self):
        self.mb_descrption_buffer.pop(self.idx)

    @property
    def cur_descrption(self) -> MicroBatchDescription:
        return self.mb_descrption_buffer.get(self.idx)

    @property
    def cur_kv_cache(self):
        if self.cur_descrption is None:
            return None
        return self.cur_descrption.kv_cache

    @property
    def cur_state(self):
        """
        Return the state of current micro batch, when current descrption is None, the state is PREFILL

        """
        if self.cur_descrption is None:
            return Status.PREFILL
        return self.cur_descrption.state
