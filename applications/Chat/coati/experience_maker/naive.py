import torch
import torch.nn.functional as F
from coati.models.base import Actor, Critic, RewardModel
from coati.models.generation import generate
from coati.models.utils import calc_action_log_probs, compute_reward
from transformers import PreTrainedTokenizer

from .base import Experience, ExperienceMaker


class NaiveExperienceMaker(ExperienceMaker):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        reward_model: RewardModel,
        initial_model: Actor,
        tokenizer: PreTrainedTokenizer,
        kl_coef: float = 0.01,
        gamma: float = 1.0,
        lam: float = 0.95,
    ) -> None:
        super().__init__(actor, critic, reward_model, initial_model)
        self.tokenizer = tokenizer
        self.kl_coef = kl_coef
        self.gamma = gamma
        self.lam = lam

    @torch.no_grad()
    def calculate_advantage(self, reward: torch.Tensor, value: torch.Tensor, num_actions: int) -> torch.Tensor:
        lastgaelam = 0
        advantages_reversed = []
        for t in reversed(range(num_actions)):
            nextvalues = value[:, t + 1] if t < num_actions - 1 else 0.0
            delta = reward[:, t] + self.gamma * nextvalues - value[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        return advantages

    @torch.no_grad()
    def make_experience(self, input_ids: torch.Tensor, **generate_kwargs) -> Experience:
        self.actor.eval()
        self.critic.eval()
        self.initial_model.eval()
        self.reward_model.eval()

        # generate sequences

        sequences = generate(self.actor, input_ids, self.tokenizer, **generate_kwargs)

        # calculate auxiliary tensors
        attention_mask = None
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is not None:
            attention_mask = sequences.not_equal(pad_token_id).to(dtype=torch.long, device=sequences.device)

        input_len = input_ids.size(1)
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            action_mask = torch.ones_like(sequences, dtype=torch.bool)
        else:
            # left padding may be applied, only mask action
            action_mask = (sequences[:, input_len:] == eos_token_id).cumsum(dim=-1) == 0
            action_mask = F.pad(action_mask, (1 + input_len, -1), value=True)  # include eos token and input
        action_mask[:, :input_len] = False
        action_mask = action_mask[:, 1:]
        action_mask = action_mask[:, -(sequences.size(1) - input_len) :]
        num_actions = action_mask.size(1)

        actor_output = self.actor(sequences, attention_mask)["logits"]
        action_log_probs = calc_action_log_probs(actor_output, sequences, num_actions)

        base_model_output = self.initial_model(sequences, attention_mask)["logits"]

        base_action_log_probs = calc_action_log_probs(base_model_output, sequences, num_actions)
        value = self.critic(sequences, attention_mask)

        # convert from left_padding, prompt, answer, right_padding to prompt, answer, right_padding
        sequences_rm = {}
        reversed_sequences = sequences.flip(dims=[1])
        indices = (
            torch.arange(1, sequences.size(1) + 1, requires_grad=False, device=sequences.device)
            .unsqueeze(0)
            .tile((sequences.size(0), 1))
        )
        mask = reversed_sequences != pad_token_id
        _, sorted_indices = (indices * mask).sort(dim=1, descending=False)
        sorted_sequences = torch.gather(reversed_sequences, 1, sorted_indices)
        sequences_rm["input_ids"] = sorted_sequences.flip(dims=[1])
        sequences_rm["attention_mask"] = sequences_rm["input_ids"] != pad_token_id

        r = self.reward_model(
            sequences=sequences_rm["input_ids"].to(dtype=torch.long, device=sequences.device),
            attention_mask=sequences_rm["attention_mask"].to(device=sequences.device),
        )
        reward, kl = compute_reward(r, self.kl_coef, action_log_probs, base_action_log_probs, action_mask=action_mask)

        value = value[:, -num_actions:] * action_mask

        advantages = self.calculate_advantage(reward, value, num_actions)

        advantages = advantages.detach()
        value = value.detach()
        r = r.detach()

        return Experience(sequences, action_log_probs, value, r, kl, advantages, attention_mask, action_mask)
