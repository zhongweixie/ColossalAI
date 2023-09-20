from typing import Optional

import torch
import torch.nn as nn

from ..lora import LoRAModule


class RewardModel(LoRAModule):
    """
    Reward model base class.

    Args:
        model (nn.Module): Reward model.
        value_head (nn.Module): Value head to get reward score.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(
        self,
        model: nn.Module,
        value_head: Optional[nn.Module] = None,
        lora_rank: int = 0,
        lora_train_bias: str = "none",
    ) -> None:
        super().__init__(lora_rank=lora_rank, lora_train_bias=lora_train_bias)
        self.model = model
        self.convert_to_lora()

        if value_head is not None:
            if value_head.out_features != 1:
                raise ValueError("The value head of reward model's output dim should be 1!")
            self.value_head = value_head
        else:
            self.value_head = nn.Linear(model.config.n_embd, 1)

    def forward(self, sequences: torch.LongTensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(sequences, attention_mask=attention_mask)
        last_hidden_states = outputs["last_hidden_state"]
        sequence_lengths = torch.max(attention_mask * torch.arange(sequences.size(1), device=sequences.device), dim=1)[
            0
        ]
        sequence_hidden_states = last_hidden_states[torch.arange(last_hidden_states.size(0)), sequence_lengths]
        values = self.value_head(sequence_hidden_states).squeeze(1)  # ensure shape is (B, )
        return values
