from numpy import allclose
import torch
from torch import nn
from colossalai.gemini.tensor.stateful_tensor import StatefulTensorV2
# TODO(jiaruifang) auto import
from colossalai.gemini.tensor._ops import *
from colossalai.gemini.tensor.api import _STATEFUL_OPS
from copy import deepcopy


def test_linear():
    in_dim = 4
    out_dim = 5

    fc = torch.nn.Linear(in_dim, out_dim, bias=True)
    fc_ref = deepcopy(fc)

    input_ref = torch.randn(1, in_dim)
    input = input_ref.clone()

    sharded_weight = StatefulTensorV2(fc_ref.weight)
    sharded_bias = StatefulTensorV2(fc_ref.bias)

    # replace the torch nn.Parameters with ShardedTensor
    delattr(fc, 'weight')
    setattr(fc, 'weight', sharded_weight)
    delattr(fc, 'bias')
    setattr(fc, 'bias', sharded_bias)

    fc.weight.requires_grad = True
    fc.bias.requires_grad = True

    # torch.nn.functional.linear(torch.randn(1, in_dim), sharded_weight, sharded_bias)
    out = fc(input)
    loss = out.sum()
    loss.backward()

    out_ref = fc_ref(input_ref)
    loss_ref = out_ref.sum()
    loss_ref.backward()

    assert (loss_ref == loss)
    assert allclose(fc_ref.weight.grad, fc.weight.torch_tensor().grad)


# The test case failed
# def test_uniform():
#     t = StatefulTensorV2(torch.zeros(3, 5))
#     # print(_STATEFUL_OPS)
#     torch.nn.init.uniform_(t)
#     print(t)


def test_element_wise():
    t_ref = torch.randn(3, 5)
    t = StatefulTensorV2(t_ref.clone())
    assert torch.mean(t) == torch.mean(t_ref)
    assert allclose(torch.nn.functional.gelu(t), torch.nn.functional.gelu(t_ref))
    assert allclose(torch.nn.functional.relu(t), torch.nn.functional.relu(t_ref))


if __name__ == '__main__':
    test_linear()
    # test_element_wise()
