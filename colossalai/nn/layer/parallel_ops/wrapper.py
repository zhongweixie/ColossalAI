# The functions used in this file are copied from pytorch

from typing import Dict, Callable
import functools

# Custom sharded ops
_SHARDED_OPS: Dict[str, Callable] = {}


def _register_sharded_op(op, func):
    from inspect import signature
    if len(signature(func).parameters) != 4:
        raise TypeError(f'Custom sharded op function expects signature: '
                        f'(types, args, kwargs, process_group), but received '
                        f'signature: {signature(func)}')

    global _SHARDED_OPS
    _SHARDED_OPS[op] = func


def sharded_op_impl(func):
    """
    Provides a way for users to write their own custom sharded operator. This
    can be used to override existing ShardedTensor operators or write a new
    one not supported by ShardedTensor. If the operator in question is covered
    by ``__torch_function__`` dispatch and has a ShardedTensor as any of its
    parameters, the function provided will be invoked for that operator.

    Example::
        >>> @custom_sharded_op(torch.nn.functional.linear)
        >>> def my_custom_sharded_linear(types, args, kwargs, process_group):
        >>>   ....
        >>>
        >>> input = torch.rand(10, 32)
        >>> weight = sharded_tensor.rand(32, 16)
        >>> bias = torch.rand(16)
        >>> # This will call 'my_custom_sharded_linear'
        >>> torch.nn.functional.linear(input, weight, bias)

    The types, args and kwargs parameters are the same parameters that are
    passed to ``__torch_function__`` dispatch API
    (https://pytorch.org/docs/stable/notes/extending.html#extending-torch).
    There is an additional ``process_group`` parameter which is the
    process_group used for the ShardedTensor and can be used by
    implementations for communications within a sharded implementation.

    Args:
        func(Callable): Torch function for which we want to provide a sharded
            implementation (ex: torch.nn.functional.linear)
    """

    def decorator_sharded_func(wrapped_func):
        _register_sharded_op(func, wrapped_func)

        @functools.wraps(wrapped_func)
        def wrapper(*args, **kwargs):
            return wrapped_func(*args, **kwargs)

        return wrapper

    return decorator_sharded_func
