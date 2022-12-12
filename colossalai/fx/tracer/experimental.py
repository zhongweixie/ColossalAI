import enum
import functools
import inspect
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
from torch.fx import Graph, Node, Proxy, Tracer
from torch.utils._pytree import tree_map

from colossalai.fx import ColoGraphModule, compatibility, is_compatible_with_meta
from colossalai.fx.tracer._tracer_utils import extract_meta, is_element_in_list
from colossalai.fx.tracer.registry import (
    bias_addition_function,
    bias_addition_method,
    bias_addition_module,
    meta_patched_function,
    meta_patched_module,
)

if is_compatible_with_meta():
    from colossalai.fx.profiler import MetaTensor

Target = Union[Callable[..., Any], str]
Argument = Optional[Union[Tuple[Any, ...],    # actually Argument, but mypy can't represent recursive types
                          List[Any],    # actually Argument
                          Dict[str, Any],    # actually Argument
                          slice,    # Slice[Argument, Argument, Argument], but slice is not a templated type in typing
                          'Node',]]
_CScriptMethod = ['add', 'mul', 'sub', 'div']
_TorchNewMethod = [
    "arange", "zeros", "zeros_like", "ones", "ones_like", "full", "full_like", "empty", "empty_like", "eye", "tensor",
    "finfo"
]
_TensorPropertyMethod = ["dtype", "shape", "device", "requires_grad", "grad", "grad_fn", "data"]


def _truncate_suffix(s: str):
    import re
    return re.sub(r'_\d+$', '', s)


def default_device():
    return torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


@compatibility(is_backward_compatible=False)
class ColoProxy(Proxy):

    def __init__(self, *args, data=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._data = data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, args):
        wrap_fn = lambda x: MetaTensor(x) if isinstance(x, torch.Tensor) else x
        self._data = tree_map(wrap_fn, args)

    @classmethod
    def __torch_function__(cls, orig_method, types, args=(), kwargs=None):
        proxy = cls.from_torch_proxy(super().__torch_function__(orig_method, types, args, kwargs))
        unwrap_fn = lambda p: p.data if isinstance(p, ColoProxy) else p
        kwargs = {} if kwargs is None else kwargs
        if proxy.data is None:
            proxy.data = orig_method(*tree_map(unwrap_fn, args), **tree_map(unwrap_fn, kwargs))
        return proxy

    @classmethod
    def from_torch_proxy(cls, proxy: Proxy):
        return cls(proxy.node, proxy.tracer)

    def __repr__(self):
        return f"ColoProxy({self.node.name}, data={self.data})"

    def __len__(self):
        return len(self.data)

    def __int__(self):
        return int(self.data)

    def __index__(self):
        try:
            return int(self.data)
        except:
            return torch.zeros(self.data.shape, dtype=torch.bool).numpy().__index__()

    def __float__(self):
        return float(self.data)

    def __bool__(self):
        return self.data

    def __getattr__(self, k):
        return ColoAttribute(self, k, getattr(self._data, k, None))

    def __contains__(self, key):
        if self.node.op == "placeholder":
            # this is used to handle like
            # if x in kwargs
            # we don't handle this case for now
            return False
        return super().__contains__(key)

    def __isinstancecheck__(self, type):
        return isinstance(self.data, type)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def device(self):
        proxy = self.tracer.create_proxy('call_function', getattr, (self, 'device'), {})
        proxy.data = self.data.device
        return proxy

    @property
    def dtype(self):
        proxy = self.tracer.create_proxy('call_function', getattr, (self, 'dtype'), {})
        proxy.data = self.data.dtype
        return proxy

    def to(self, *args, **kwargs):
        return self.tracer.create_proxy('call_method', 'to', (self, *args), {**kwargs})

    def cpu(self, *args, **kwargs):
        return self.tracer.create_proxy('call_method', 'cpu', (self, *args), {**kwargs})

    def cuda(self, *args, **kwargs):
        return self.tracer.create_proxy('call_method', 'cuda', (self, *args), {**kwargs})


@compatibility(is_backward_compatible=False)
class ColoAttribute(ColoProxy):

    def __init__(self, root, attr: str, data=None):
        self.root = root
        self.attr = attr
        self.tracer = root.tracer
        self._data = data
        self._node: Optional[Node] = None

    @property
    def node(self):
        # the node for attributes is added lazily, since most will just be method calls
        # which do not rely on the getitem call
        if self._node is None:
            self._node = self.tracer.create_proxy('call_function', getattr, (self.root, self.attr), {}).node
        return self._node

    def __call__(self, *args, **kwargs):
        return self.tracer.create_proxy('call_method', self.attr, (self.root,) + args, kwargs)

    def __repr__(self):
        return f"ColoAttribute({self.node.name}, attr={self.attr})"


@compatibility(is_backward_compatible=False)
class ColoTracer(Tracer):

    def __init__(self, trace_act_ckpt: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._disable_module_getattr = False
        self.proxy_buffer_attributes = True

        # whether the tracer will record the usage of torch.utils.checkpoint
        self.trace_act_ckpt = trace_act_ckpt
        # whether the current tracing occurs within the activation checkpoint functions
        self.inside_torch_checkpoint_func = False
        self.act_ckpt_region_count = 0

    def proxy(self, node: Node) -> 'ColoProxy':
        return ColoProxy(node, self)

    def create_proxy(self,
                     kind: str,
                     target: Target,
                     args: Tuple[Any, ...],
                     kwargs: Dict[str, Any],
                     name: Optional[str] = None,
                     type_expr: Optional[Any] = None,
                     proxy_factory_fn: Callable[[Node], 'Proxy'] = None):
        proxy: ColoProxy = super().create_proxy(kind, target, args, kwargs, name, type_expr, proxy_factory_fn)
        # unwrap_fn = lambda p: p.data if isinstance(p, ColoProxy) else p
        # if kind == 'placeholder':
        #     proxy.data = self.meta_args[target] if target in self.meta_args else self.concrete_args.get(
        #         _truncate_suffix(target), None)
        # elif kind == 'get_attr':
        #     self._disable_module_getattr = True
        #     try:
        #         attr_itr = self.root
        #         atoms = target.split(".")
        #         for atom in atoms:
        #             attr_itr = getattr(attr_itr, atom)
        #         proxy.data = attr_itr
        #     finally:
        #         self._disable_module_getattr = False
        # elif kind == 'call_function':
        #     proxy.data = target(*tree_map(unwrap_fn, args), **tree_map(unwrap_fn, kwargs))
        # elif kind == 'call_method':
        #     self._disable_module_getattr = True
        #     try:
        #         if target == '__call__':
        #             proxy.data = unwrap_fn(args[0])(*tree_map(unwrap_fn, args[1:]), **tree_map(unwrap_fn, kwargs))
        #         else:
        #             if target not in _TensorPropertyMethod:
        #                 proxy._data = getattr(unwrap_fn(args[0]), target)(*tree_map(unwrap_fn, args[1:]),
        #                                                                   **tree_map(unwrap_fn, kwargs))
        #     finally:
        #         self._disable_module_getattr = False
        # elif kind == 'call_module':
        #     mod = self.root.get_submodule(target)
        #     unwrap_fn = lambda p: p.data if isinstance(p, ColoProxy) else p
        #     self._disable_module_getattr = True
        #     try:
        #         proxy.data = mod.forward(*tree_map(unwrap_fn, args), **tree_map(unwrap_fn, kwargs))
        #     finally:
        #         self._disable_module_getattr = True
        return proxy

    def create_node(self, *args, **kwargs) -> Node:
        node = super().create_node(*args, **kwargs)

        if self.inside_torch_checkpoint_func:
            # annotate the activation checkpoint module
            node.meta['activation_checkpoint'] = self.act_ckpt_region_count
        return node

    def trace(self,
              root: torch.nn.Module,
              concrete_args: Optional[Dict[str, torch.Tensor]] = None,
              meta_args: Optional[Dict[str, torch.Tensor]] = None) -> Graph:

        if meta_args is None:
            meta_args = {}

        if concrete_args is None:
            concrete_args = {}

        # check concrete and meta args have valid names
        sig = inspect.signature(root.forward)
        sig_names = set(sig.parameters.keys())
        meta_arg_names = set(meta_args.keys())

        # update concrete args with default values
        non_meta_arg_names = sig_names - meta_arg_names
        for k, v in sig.parameters.items():
            if k in non_meta_arg_names and \
                    k not in concrete_args and \
                    v.default is not inspect.Parameter.empty:
                concrete_args[k] = v.default

        # get non concrete arg names
        concrete_arg_names = set(concrete_args.keys())
        non_concrete_arg_names = sig_names - concrete_arg_names

        def _check_arg_name_valid(names):
            success, element = is_element_in_list(names, sig_names)
            if not success:
                raise KeyError(
                    f"argument {element} is not found in the signature of {root.__class__.__name__}'s forward function")

        _check_arg_name_valid(meta_arg_names)
        _check_arg_name_valid(concrete_arg_names)

        self.concrete_args = concrete_args
        self.meta_args = meta_args

        with _TorchTensorOverride(self), self.trace_activation_checkpoint(enabled=self.trace_act_ckpt):
            self.graph = super().trace(root, concrete_args=concrete_args)
        self.graph.lint()
        return self.graph


    @contextmanager
    def trace_activation_checkpoint(self, enabled: bool):
        if enabled:
            orig_ckpt_func = torch.utils.checkpoint.CheckpointFunction

            class PatchedCheckpointFunction(torch.autograd.Function):

                @staticmethod
                def forward(ctx, run_function, preserve_rng_state, *args):
                    # signal that the current tracing occurs within activaton checkpoint part
                    self.inside_torch_checkpoint_func = True
                    out = run_function(*args)
                    self.inside_torch_checkpoint_func = False
                    self.act_ckpt_region_count += 1
                    return out

                @staticmethod
                def backward(ctx: Any, *grad_outputs: Any) -> Any:
                    raise NotImplementedError(
                        "We do not implement the backward pass as we only trace the forward pass.")

            # override the checkpoint function
            torch.utils.checkpoint.CheckpointFunction = PatchedCheckpointFunction
        yield

        if enabled:
            # recover the checkpoint function upon exit
            torch.utils.checkpoint.CheckpointFunction = orig_ckpt_func


    def _post_check(self, non_concrete_arg_names: Set[str]):
        # This is necessary because concrete args are added as input to the traced module since
        # https://github.com/pytorch/pytorch/pull/55888.
        for node in self.graph.nodes:
            if node.op == "placeholder":
                # Removing default values for inputs as the forward pass will fail with them.
                if node.target in non_concrete_arg_names:
                    node.args = ()
                    # Without this, torch.jit.script fails because the inputs type is Optional[torch.Tensor].
                    # It cannot infer on the attributes and methods the input should have, and fails.
                    node.type = torch.Tensor
                # It is a concrete arg so it is not used and should be removed.
                else:
                    if hasattr(torch.fx._symbolic_trace, "_assert_is_none"):
                        # Newer versions of torch.fx emit an assert statement
                        # for concrete arguments; delete those before we delete
                        # the concrete arg.
                        to_delete = []
                        for user in node.users:
                            if user.target == torch.fx._symbolic_trace._assert_is_none:
                                to_delete.append(user)
                        for user in to_delete:
                            self.graph.erase_node(user)

                    self.graph.erase_node(node)

            # TODO: solves GraphModule creation.
            # Without this, return type annotation "Tuple" is causing code execution failure.
            if node.op == "output":
                node.type = None
            self.graph.lint()

    def _module_getattr(self, attr, attr_val, parameter_proxy_cache):
        if getattr(self, "_disable_module_getattr", False):
            return attr_val

        def maybe_get_proxy_for_attr(attr_val, collection_to_search, parameter_proxy_cache):
            for n, p in collection_to_search:
                if attr_val is p:
                    if n not in parameter_proxy_cache:
                        kwargs = {}
                        if 'proxy_factory_fn' in inspect.signature(self.create_proxy).parameters:
                            kwargs['proxy_factory_fn'] = (None if not self.param_shapes_constant else
                                                          lambda node: ColoProxy(self, node, n, attr_val))
                        val_proxy = self.create_proxy('get_attr', n, (), {}, **kwargs)    # type: ignore[arg-type]
                        parameter_proxy_cache[n] = val_proxy
                    return parameter_proxy_cache[n]
            return None

        if self.proxy_buffer_attributes and isinstance(attr_val, torch.Tensor):
            maybe_buffer_proxy = maybe_get_proxy_for_attr(attr_val, self.root.named_buffers(), parameter_proxy_cache)
            if maybe_buffer_proxy is not None:
                return maybe_buffer_proxy

        if isinstance(attr_val, torch.nn.Parameter):
            maybe_parameter_proxy = maybe_get_proxy_for_attr(attr_val, self.root.named_parameters(),
                                                             parameter_proxy_cache)
            if maybe_parameter_proxy is not None:
                return maybe_parameter_proxy

        return attr_val


@compatibility(is_backward_compatible=True)
def symbolic_trace(
    root: Union[torch.nn.Module, Callable[..., Any]],
    concrete_args: Optional[Dict[str, Any]] = None,
    meta_args: Optional[Dict[str, Any]] = None,
) -> ColoGraphModule:
    if is_compatible_with_meta():
        if meta_args is not None:
            root.to(default_device())
            wrap_fn = lambda x: MetaTensor(x, fake_device=default_device()) if isinstance(x, torch.Tensor) else x
            graph = ColoTracer().trace(root, concrete_args=concrete_args, meta_args=tree_map(wrap_fn, meta_args))
            root.cpu()
        else:
            graph = Tracer().trace(root, concrete_args=concrete_args)
    else:
        from .tracer import ColoTracer as OrigColoTracer
        graph = OrigColoTracer().trace(root, concrete_args=concrete_args, meta_args=meta_args)
    name = root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    return ColoGraphModule(root, graph, name)


@compatibility(is_backward_compatible=False)
class _TorchTensorOverride(object):

    def __init__(self, tracer: Tracer):
        self.overrides = {}
        self.tracer = tracer

    def __enter__(self):

        def wrap_tensor_method(target):

            @functools.wraps(target)
            def wrapper(*args, **kwargs):
                is_proxy = any(isinstance(p, ColoProxy) for p in args) | any(
                    isinstance(p, ColoProxy) for p in kwargs.values())
                if is_proxy:
                    # if the arg is a proxy, then need to record this function called on this proxy
                    # e.g. torch.ones(size) where size is an input proxy
                    self.tracer._disable_module_getattr = True
                    try:
                        proxy = self.tracer.create_proxy('call_function', target, args, kwargs)
                    finally:
                        self.tracer._disable_module_getattr = False
                    return proxy
                else:
                    return target(*args, **kwargs)

            return wrapper, target

        self.overrides = {
            target: wrap_tensor_method(getattr(torch, target))
            for target in _TorchNewMethod
            if callable(getattr(torch, target))
        }
        for name, (wrapper, orig) in self.overrides.items():
            setattr(torch, name, wrapper)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, (wrapper, orig) in self.overrides.items():
            setattr(torch, name, orig)


def set_meta_data_for_node(gm: ColoGraphModule, root: torch.nn.Module, meta_args: Optional[Dict[str, Any]]=None):
    for node in gm.graph.nodes:
        node._meta_data = _meta_data_computing(meta_args, root, node.op, node.target, node.args, node.kwargs)

def _meta_data_computing(meta_args, root, kind, target, args, kwargs):
    if kind == "placeholder" and target in meta_args and meta_args[target].is_meta:
        meta_out = meta_args[target]
        return meta_out

    if callable(getattr(torch, target)) and target.__name__ in _TorchNewMethod:
        # NOTE: tensor constructors in PyTorch define the `device` argument as
        # *kwargs-only*. That is why this works. If you add methods to
        # _TORCH_METHODS_TO_PATCH that do not define `device` as kwarg-only,
        # this will break and you will likely see issues where we cannot infer
        # the size of the output.
        if "device" in kwargs:
            kwargs["device"] = "meta"

    try:
        args_metas, kwargs_metas = extract_meta(*args, **kwargs)
        unwrap_fn = lambda p: p._meta_data if isinstance(p, Node) else p

        if kind == "call_function":
            # fetch patched function
            if meta_patched_function.has(target):
                meta_target = meta_patched_function.get(target)
            elif meta_patched_function.has(target.__name__):
                # use name for some builtin op like @ (matmul)
                meta_target = meta_patched_function.get(target.__name__)
            else:
                meta_target = target

            meta_out = meta_target(*tree_map(unwrap_fn, args_metas), **tree_map(unwrap_fn, kwargs_metas))

            if isinstance(meta_out, torch.Tensor):
                meta_out = meta_out.to(device="meta")
        elif kind == "call_method":
            method = getattr(args_metas[0].__class__, target)

            # fetch patched method
            if meta_patched_function.has(method):
                meta_target = meta_patched_function.get(method)
            else:
                meta_target = method

            meta_out = meta_target(*tree_map(unwrap_fn, args_metas), **tree_map(unwrap_fn, kwargs_metas))
        elif kind == "call_module":
            mod = root.get_submodule(target)
            mod_type = type(mod)
            if meta_patched_module.has(mod_type):
                meta_out = meta_patched_module.get(mod_type)(mod, *tree_map(unwrap_fn, args_metas),
                                                             **tree_map(unwrap_fn, kwargs_metas))
            else:
                meta_out = mod(*tree_map(unwrap_fn, args_metas), **tree_map(unwrap_fn, kwargs_metas))
        elif kind == "get_attr":
            attr_itr = root
            atoms = target.split(".")
            for atom in atoms:
                attr_itr = getattr(attr_itr, atom)
            if isinstance(attr_itr, torch.nn.parameter.Parameter):
                meta_out = torch.nn.Parameter(attr_itr.to(device="meta"))
            elif isinstance(attr_itr, torch.Tensor):
                meta_out = attr_itr.to(device="meta")
            else:
                meta_out = attr_itr
        else:
            return None

    except Exception as e:
        raise RuntimeError(f"Could not compute metadata for {kind} target {target}: {e}")

    return meta_out
