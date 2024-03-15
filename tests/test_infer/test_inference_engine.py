import random

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, GenerationConfig, LlamaConfig, LlamaForCausalLM

import colossalai
from colossalai.inference.config import _DEFAULT_PROMPT_TEMPLATES, InferenceConfig
from colossalai.inference.core.engine import InferenceEngine
from colossalai.inference.flash_decoding_utils import FDIntermTensors
from colossalai.inference.modeling.models.glide_llama import GlideLlamaConfig
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def check_inference_engine(use_engine=False, prompt_template=None):
    setup_seed(20)
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    model = (
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=50000, hidden_size=512, intermediate_size=1536, num_attention_heads=4, num_hidden_layers=16
            )
        )
        .cuda()
        .half()
    )
    model = model.eval()

    inputs = [
        "介绍一下今天的北京,比如故宫，天安门，长城或者其他的一些景点,",
        "介绍一下武汉,",
    ]

    output_len = 38
    do_sample = True
    top_p = 0.5
    top_k = 50

    if use_engine:
        inference_config = InferenceConfig(max_output_len=output_len, prompt_template=prompt_template)
        inference_engine = InferenceEngine(model, tokenizer, inference_config, verbose=True)
        assert inference_engine.generation_config.max_new_tokens == output_len
        inference_engine.add_request(prompts=inputs)
        assert inference_engine.request_handler._has_waiting()
        generation_config = GenerationConfig(do_sample=do_sample, top_p=top_p, top_k=top_k)
        outputs = inference_engine.generate(generation_config=generation_config)
    else:
        if prompt_template:
            # apply prompt template
            inputs = [_DEFAULT_PROMPT_TEMPLATES[prompt_template].format(input_text=input_text) for input_text in inputs]
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        inputs = tokenizer.batch_encode_plus(inputs, padding=True, return_tensors="pt")["input_ids"]
        inputs = inputs.cuda()
        generation_config = GenerationConfig(
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=output_len,
        )
        outputs = model.generate(inputs, generation_config=generation_config)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return outputs


@parameterize("prompt_template", [None, "llama"])
def check_output_consistency(prompt_template):
    cai_outputs = check_inference_engine(use_engine=True, prompt_template=prompt_template)
    transformer_outputs = check_inference_engine(use_engine=False, prompt_template=prompt_template)

    for s1, s2 in zip(cai_outputs, transformer_outputs):
        assert s1 == s2, f"\nColossalAI Output: {s1}\nTransformers Output: {s2}"

    # clear singleton flash decoding tensors
    FDIntermTensors._instances = {}


@parameterize("num_layers", [1])
@parameterize("max_length", [100])
def check_spec_dec(num_layers, max_length):
    torch.manual_seed(123)

    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    # Dummy configs for testing
    toy_config = LlamaConfig(num_hidden_layers=num_layers)
    toy_config.pad_token_id = tokenizer.eos_token_id
    drafter_model = LlamaForCausalLM(toy_config)
    drafter_model = drafter_model.eval().cuda()
    large_config = LlamaConfig(
        hidden_size=4096,
        intermediate_size=11008,
        num_attention_heads=32,
        num_hidden_layers=8,
        num_key_value_heads=32,
        max_position_embeddings=2048,
    )
    large_config.pad_token_id = tokenizer.eos_token_id
    main_model = LlamaForCausalLM(large_config)

    inference_config = InferenceConfig(
        dtype="fp16",
        micro_batch_size=1,
        max_batch_size=1,
        max_input_len=128,
        max_output_len=128,
        prefill_ratio=1.2,
        block_size=16,
    )
    engine = InferenceEngine(main_model, tokenizer, inference_config)
    engine.enable_spec_dec(drafter_model, n_spec_tokens=5)

    dummy_inputs = torch.randint(low=5, high=1000, size=(1, 10), dtype=torch.long, device="cuda")
    generation_config = GenerationConfig(
        pad_token_id=tokenizer.eos_token_id,
        max_length=max_length,
        eos_token_id=tokenizer.eos_token_id,
    )
    out, out_token_ids = engine.generate(
        prompts_token_ids=dummy_inputs, generation_config=generation_config, return_token_ids=True
    )
    engine.disable_spec_dec()
    engine.clear_spec_dec()

    assert not engine.use_spec_dec
    assert engine.drafter is None and engine.drafter_model is None

    assert len(out) == 1
    assert len(out_token_ids) == 1 and len(out_token_ids[0]) == max_length

    glide_config = GlideLlamaConfig(
        intermediate_size=8192,
        large_hidden_size=4096,
        large_num_attention_heads=32,
        num_hidden_layers=1,
    )
    drafter_model = LlamaForCausalLM(glide_config)
    glide_model = engine.convert_to_glide_model(drafter_model, drafter_model.state_dict())  # dummy state dict

    engine.enable_spec_dec(glide_model, use_glide_drafter=True)

    out, out_token_ids = engine.generate(
        prompts_token_ids=dummy_inputs, generation_config=generation_config, return_token_ids=True
    )
    engine.clear_spec_dec()

    assert len(out) == 1
    assert len(out_token_ids) == 1 and len(out_token_ids[0]) == max_length


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, port=port, host="localhost")
    check_output_consistency()
    check_spec_dec()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_inference_engine():
    spawn(run_dist, 1)


if __name__ == "__main__":
    test_inference_engine()
