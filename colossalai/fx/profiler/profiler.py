from dataclasses import dataclass
from enum import auto
from typing import Callable, Any, Dict, Tuple
import torch
from torch.fx import Graph, Node
from torch.fx.node import Argument, Target
from torch.utils._pytree import tree_map
from .dataflow import autograd_graph_analysis, Stage
from .memory import WEIRD_OPS, activation_size
from .tensor import MetaTensor
from .opcount import flop_mapping

__all__ = ['profile_function', 'profile_module', 'profile_method']


def normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


def is_autogradable(x):
    return isinstance(x, torch.Tensor) and x.is_floating_point()


@dataclass
class MetaInfo:
    """
    This is a dataclass for MetaInfo, which measures
    the execution memory cost and FLOPs with `MetaTensor`.
    Attributes:
        fwd_flop (int): The forward FLOPs of a certain node
        bwd_flop (int): The backward FLOPs of a certain node.
        fwd_in (int): See definitions in https://github.com/hpcaitech/ColossalAI/tree/main/colossalai/fx/profiler/dataflow.py
        fwd_tmp (int): See definitions in https://github.com/hpcaitech/ColossalAI/tree/main/colossalai/fx/profiler/dataflow.py
        bwd_tmp (int): See definitions in https://github.com/hpcaitech/ColossalAI/tree/main/colossalai/fx/profiler/dataflow.py
        bwd_out (int): See definitions in https://github.com/hpcaitech/ColossalAI/tree/main/colossalai/fx/profiler/dataflow.py
    """
    fwd_flop: int = 0
    bwd_flop: int = 0
    fwd_in: int = 0
    fwd_tmp: int = 0
    bwd_tmp: int = 0
    bwd_out: int = 0


def _profile(target: Callable, *args, inplace=False, **kwargs) -> Tuple[Any, ...]:
    """Profile a Callable function with args and kwargs.

    Args:
        target (Callable): A Callable function
        args (Any): Argument
        kwargs (Any): Argument

    Returns:
        out (Tuple[Any, ...]): The argument value that was retrieved.
        meta_info (MetaInfo): The memory cost and FLOPs estimated with `MetaTensor`.
    """
    # This subgraph traces aten level ops inside one node.
    subgraph = Graph()

    meta_info = MetaInfo()

    # `flop_count`` serves as a global dictionary to store results.
    flop_count = {
        Stage.F: 0,
        Stage.L: 0,
        Stage.B: 0,
    }

    # `stage` will mark the stage of autograd from outside scope.
    stage = Stage.F

    # FlopTensor not only get the flop statistics of a single node,
    # it also build a full autograd graph for this node.
    # This makes sure we can analyze the dependencies of memory, and
    # decide which forward intermediate results should be kept until
    # backward is executed.
    # Hopefully, this attempt will provide a better estimation of memory.
    class FlopTensor(MetaTensor):

        _node: Node

        def __repr__(self):
            if self.grad_fn:
                return f"FlopTensor(..., device={self._tensor.device}, size={tuple(self.shape)}, grad_fn={self.grad_fn})"
            return f"FlopTensor(..., device={self._tensor.device}, size={tuple(self.shape)})"

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):

            def get_node(x):
                return None if not hasattr(x, '_node') else x._node

            args_node = tree_map(get_node, args)
            kwargs_node = tree_map(get_node, kwargs)
            node = subgraph.create_node('call_function', func, args_node, kwargs_node)

            def unwrap(x):
                # if x is a `nn.Parameter`, we can first wrap it with `FlopTensor`
                if isinstance(x, torch.Tensor) and not hasattr(x, '_tensor'):
                    x = FlopTensor(x.to('meta'))
                return x._tensor.to('meta') if isinstance(x, FlopTensor) else x

            args = tree_map(unwrap, args)
            kwargs = tree_map(unwrap, kwargs)

            # run aten for backend=CPU but actually on backend=Meta
            out = func(*args, **kwargs)
            flop_count[stage] += flop_mapping[func](args, normalize_tuple(out))
            node.meta['out'] = normalize_tuple(out)
            node.meta['stage'] = stage

            def wrap(x):
                return FlopTensor(x.to('meta')) if isinstance(x, torch.Tensor) else x

            def set_node(x):
                x._node = node

            out = tree_map(wrap, out)
            tree_map(set_node, out)
            return out

    # `WEIRD_OPS` are tough to handle because they don't accept autograd
    #  on meta tensor.
    if target not in WEIRD_OPS:

        def wrap(x):
            return FlopTensor(x.detach().requires_grad_(
                True)) if is_autogradable(x) and not inplace and not hasattr(x, '_tensor') else x
    else:

        def wrap(x):
            return FlopTensor(x.detach().requires_grad_(
                False)) if is_autogradable(x) and not inplace and not hasattr(x, '_tensor') else x

    # Basically, we need to detach the args and kwargs from the outer graph.
    args = tree_map(wrap, args)
    kwargs = tree_map(wrap, kwargs)

    def set_placeholder(x):
        if isinstance(x, FlopTensor):
            x._node = subgraph.create_node('placeholder',
                                           'placeholder', (subgraph._root,),
                                           name=subgraph._graph_namespace.create_name('input', x._tensor))
            x._node.meta['stage'] = Stage.P
            x._node.meta['out'] = (x._tensor,)

    tree_map(set_placeholder, args)
    tree_map(set_placeholder, kwargs)

    def pack(x):
        if isinstance(x, FlopTensor):
            x._node.meta['saved'] = True
        return x

    def unpack(x):
        return x

    # mark saved tensors with saved_tensors_hooks
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        if isinstance(target, str):
            # args[0] is the `self` object for this method call
            self_obj, *args_tail = args
            out = getattr(self_obj, target)(*args_tail, **kwargs)
        else:
            out = target(*args, **kwargs)

    # If the output is not a floating point `torch.Tensor` or it does not
    # requires grad, then we should not run backward for this node.
    if is_autogradable(out) and out.requires_grad:
        stage = Stage.L
        loss = out.sum()
        stage = Stage.B
        loss.backward()

    graph_info = autograd_graph_analysis(subgraph)
    meta_info.fwd_flop, meta_info.bwd_flop = flop_count[Stage.F], flop_count[Stage.B]
    meta_info.__dict__.update(graph_info.__dict__)

    def unwrap(x):
        return x._tensor.to('meta') if isinstance(x, FlopTensor) else x

    return tree_map(unwrap, out), meta_info


def profile_function(target: 'Target') -> Callable:
    """
    Wrap a `call_function` node or `torch.nn.functional` in order to 
    record the memory cost and FLOPs of the execution.
    
    Warnings:
        You may only use tensors with `device=meta` for this wrapped function.
        Only original `torch.nn.functional` are available.
    
    Examples:
        >>> input = torch.rand(100, 100, 100, 100, device='meta')
        >>> func = torch.nn.functional.relu
        >>> output, meta_info = profile_function(func)(input, inplace=False)
    """

    def f(*args: Tuple[Argument, ...], **kwargs: Dict[str, Any]) -> Any:

        # If there is an argument that this `call_function` is inplace, we should
        # skip the autograd profiling.
        out, meta = _profile(func, *args, **kwargs)
        return out, meta

    f.__name__ = target.__name__
    func = target
    return f


def profile_method(target: 'Target') -> Callable:
    """
    Wrap a `call_method` node
    record the memory cost and FLOPs of the execution. 
    """

    def f(*args: Tuple[Argument, ...], **kwargs: Dict[str, Any]) -> Any:
        # execute the method and return the result
        assert isinstance(target, str), f'{target} instance is not str.'
        out, meta = _profile(target, *args, inplace=False, **kwargs)
        return out, meta

    return f


def profile_module(module: torch.nn.Module) -> Callable:
    """
    Wrap a `call_module` node or `torch.nn` in order to 
    record the memory cost and FLOPs of the execution.
    
    Warnings:
        You may only use tensors with `device=meta` for this wrapped function.
        Only original `torch.nn` are available.
    
    Example:
        >>> input = torch.rand(4, 3, 224, 224, device='meta')
        >>> mod = torch.nn.Conv2d(3, 128, 3)
        >>> output, meta_info = profile_module(mod)(input)
    """

    def f(*args: Tuple[Argument, ...], **kwargs: Dict[str, Any]) -> Any:

        # If there is an argument that this `call_module` is inplace, we should
        # skip the autograd profiling.
        out, meta = _profile(func, *args, inplace=getattr(module, 'inplace', False), **kwargs)
        return out, meta

    f.__name__ = module.__class__.__name__
    func = module.forward
    return f
