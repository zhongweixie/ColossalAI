import pytest
import torch
from hf_tracer_utils import trace_model_and_compare_output

from tests.kit.model_zoo import model_zoo

BATCH_SIZE = 2
SEQ_LENGTH = 16


@pytest.mark.skipif(torch.__version__ < '1.12.0', reason='torch version < 12')
def test_albert():
    sub_registry = model_zoo.get_sub_registry('transformers_albert')

    for name, (model_fn, data_gen_fn, _, _) in sub_registry.items():
        model = model_fn()
        trace_model_and_compare_output(model, data_gen_fn)


if __name__ == '__main__':
    test_albert()
