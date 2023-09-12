import pytest
import torch
from torch.testing import assert_close

import colossalai
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.zero import GeminiDDP
from colossalai.zero.gemini.chunk import search_chunk_configuration
from tests.components_to_test.registry import non_distributed_component_funcs
from tests.test_tensor.common_utils import set_seed

PLACEMENT_CONFIGS = [
    {
        'placement_policy': 'static',
        'shard_param_frac': 0.0
    },    # zero2
    {
        'placement_policy': 'static',
        'shard_param_frac': 1.0
    },    # zero3
    {
        'placement_policy': 'static',
        'shard_param_frac': 0.5
    },    # zero3-half
    {
        'placement_policy': 'auto'
    }
]


def ignore_the_first_parameter(model: torch.nn.Module):
    for name, param in model.named_parameters():
        print(f"parameter `{name}` is set ignored")
        GeminiDDP.set_params_to_ignore([param])
        return


@parameterize('placement_config', PLACEMENT_CONFIGS)
@parameterize('keep_gathered', [True, False])
@parameterize('model_name', ['gpt2', 'bert'])
def exam_state_dict(placement_config, keep_gathered, model_name: str):
    set_seed(431)
    get_components_func = non_distributed_component_funcs.get_callable(model_name)
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()

    model = model_builder()

    torch_model = model_builder()
    for torch_p, p in zip(torch_model.parameters(), model.parameters()):
        torch_p.data.copy_(p.data)

    world_size = torch.distributed.get_world_size()
    config_dict, *_ = search_chunk_configuration(model, search_range_m=1, search_interval=100)
    config_dict[world_size]['chunk_size'] = 5000
    config_dict[world_size]['keep_gathered'] = keep_gathered
    model = GeminiDDP(model, config_dict, **placement_config, pin_memory=True)
    model.train()

    zero_dict = model.state_dict(only_rank_0=False)
    torch_dict = torch_model.state_dict()

    for key, value in torch_dict.items():
        assert key in zero_dict, "{} not in ZeRO dictionary.".format(key)
        temp_zero_value = zero_dict[key].to(device=value.device, dtype=value.dtype)
        assert_close(value, temp_zero_value, rtol=1e-3, atol=1e-5)


@parameterize('placement_config', PLACEMENT_CONFIGS)
@parameterize('keep_gathered', [True, False])
@parameterize('model_name', ['gpt2', 'bert'])
def exam_load_state_dict(placement_config, keep_gathered, model_name: str):
    set_seed(431)
    get_components_func = non_distributed_component_funcs.get_callable(model_name)
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()

    model = model_builder()

    set_seed(451)
    torch_model = model_builder()    # get a different model

    world_size = torch.distributed.get_world_size()
    config_dict, *_ = search_chunk_configuration(model, search_range_m=1, search_interval=100)
    config_dict[world_size]['chunk_size'] = 5000
    config_dict[world_size]['keep_gathered'] = keep_gathered

    model = GeminiDDP(model, config_dict, **placement_config, pin_memory=True)

    torch_dict = torch_model.state_dict()
    model.load_state_dict(torch_dict, strict=False)
    zero_dict = model.state_dict(only_rank_0=False)

    for key, value in torch_dict.items():
        assert key in zero_dict, "{} not in ZeRO dictionary.".format(key)
        temp_zero_value = zero_dict[key].to(device=value.device, dtype=value.dtype)
        assert_close(value, temp_zero_value, rtol=1e-3, atol=1e-5)


@parameterize('placement_config', PLACEMENT_CONFIGS)
@parameterize('model_name', ['gpt2', 'bert'])
def exam_state_dict_shard(placement_config, model_name: str):
    get_components_func = non_distributed_component_funcs.get_callable(model_name)
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()

    model = model_builder()

    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2

    config_dict, *_ = search_chunk_configuration(model, search_range_m=1, search_interval=100)
    model = GeminiDDP(model, config_dict, **placement_config)
    model.train()

    zero_dict = model.state_dict(only_rank_0=False)
    accumulated_keys = set()
    # ensure number of shards > 1
    for shard, _ in model.state_dict_shard(max_shard_size=(model_size / 3), only_rank_0=False):
        for key, value in shard.items():
            assert key not in accumulated_keys, f"key `{key}` is duplicated."
            accumulated_keys.add(key)
            assert key in zero_dict, f"{key} not in ZeRO dictionary."
            assert torch.equal(value, zero_dict[key]), f"{key} not equal."


def run_dist(rank, world_size, port):
    config = {}
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    exam_state_dict()
    exam_load_state_dict()
    exam_state_dict_shard()


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_zero_ddp(world_size):
    spawn(run_dist, world_size)


if __name__ == '__main__':
    test_zero_ddp(1)
