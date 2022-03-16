from distutils.command.config import config
from colossalai.zero.sharded_model.sharded_model_v2 import ShardedModelV2
from colossalai.zero.sharded_optim.sharded_optim_v2 import ShardedOptimizerV2
from colossalai.zero.shard_utils import TensorShardStrategy
import torch
import torch.nn as nn
from colossalai.amp.naive_amp import NaiveAMPModel
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from torch.optim import Optimizer
from .sharded_model import ShardedModel
from .sharded_optim import ShardedOptimizer
from typing import Type


def convert_to_zero_v2(model: nn.Module, optimizer_class: Type[Optimizer],
                       **zero_config) -> (ShardedModelV2, ShardedOptimizerV2):
    """
    A helper function to integrate the model and optimizer with ZeRO optimizer and off-loading

    :param model: Your model object
    :type model: :class:`torch.nn.Module`
    :param optimizer: Your optimizer object
    :type optimizer: :class:`torch.optim.Optimizer`
    :param zero_config: Configuration for zero
    :type zero_config: dict

    :return: (model, optimizer)
    :rtype: Tuple
    """
    assert isinstance(model, nn.Module)
    #FIXME() zero init context
    zero_model = ShardedModelV2(model, shard_strategy=TensorShardStrategy())
    #FIXME()
    zero_optimizer = ShardedOptimizerV2(zero_model, optimizer_class, lr=1e-3)
    return zero_model, zero_optimizer


def convert_to_zero(model: nn.Module, optimizer: Optimizer, level: int, zero_config: dict):
    """
    A helper function to integrate the model and optimizer with ZeRO optimizer and off-loading

    :param model: Your model object
    :type model: :class:`torch.nn.Module`
    :param optimizer: Your optimizer object
    :type optimizer: :class:`torch.optim.Optimizer`
    :param level: Optimizer level, can be 2 or 3
    :type level: int
    :param zero_config: Configuration for zero
    :type zero_config: dict

    :return: (model, optimizer)
    :rtype: Tuple
    """
    assert 1 <= level <= 3, 'Only ZERO Optimizer Level 1-3 are provided'
    if level in [1, 2]:
        if level == 2:
            if 'partition_grad' in zero_config:
                assert zero_config['partition_grad'], \
                    'Sharded Optimizer requires partition_grad to be True'
            else:
                zero_config['partiton_grad'] = True
        model = NaiveAMPModel(model, output_to_fp32=True)
        optimizer = ShardedOptimizer(optimizer, **zero_config)
    else:
        model = ShardedModel(module=model, **zero_config)
    return model, optimizer


__all__ = ['convert_to_zero', 'ShardedModel', 'ShardedOptimizer']
