from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils._pytree import tree_map

from colossalai.fx.profiler.tensor import MetaTensor
from colossalai.tensor.d_tensor.d_tensor import DTensor
from colossalai.tensor.d_tensor.layout import Layout

# reference: https://pytorch.org/cppdocs/notes/tensor_creation.html
_NORMAL_FACTORY = [
    "arange",
    "empty",
    "full",
    "linspace",
    "logspace",
    "ones",
    "rand",
    "randn",
    "randint",
    "randperm",
    "zeros",
    "tensor",
]

# factory function that does not support meta tensor backend
_NO_META_FACTORY = [
    "eye",
]

_EARLY_MATERIALIZED_OPS = ['__getitem__', 'split']

# If your intent is to change the metadata of a Tensor (such as sizes / strides / storage / storage_offset)
# without autograd tracking the change, remove the .data / .detach() call and wrap the change in a `with torch.no_grad():` block.
# These ops cannot be unwrapped using .data
_CHANGE_META_OPS = ['_cudnn_rnn_flatten_weight']

_LEGACY_TENSOR_CONSTRUCTOR = {
    'FloatTensor': torch.float,
    'DoubleTensor': torch.double,
    'HalfTensor': torch.half,
    'BFloat16Tensor': torch.bfloat16,
    'ByteTensor': torch.uint8,
    'CharTensor': torch.int8,
    'ShortTensor': torch.short,
    'IntTensor': torch.int,
    'LongTensor': torch.long,
    'BoolTensor': torch.bool,
}

_EMPTY_DATA = torch.empty(0)


class _MyTensor(Tensor):
    """This class is only for correctness verification.
    """
    _pre_op_fn: Callable[['LazyTensor'], None] = lambda *args: None

    def __new__(cls, func, *args, concrete_data=None, **kwargs) -> '_MyTensor':
        cls._pre_op_fn()
        if concrete_data is not None:
            # uniform api as LazyTensor
            data = concrete_data
        else:
            data = func(*args, **kwargs)
        return Tensor._make_subclass(cls, data, require_grad=data.requires_grad)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        cls._pre_op_fn()
        return super().__torch_function__(func, types, args, kwargs)


class LazyTensor(torch.Tensor):
    """A naive implementation of LazyTensor (https://arxiv.org/pdf/2102.13267.pdf).

    Usage:
        1. Use ``LazyTensor`` instead of ``torch.Tensor``.
        >>> x = LazyTensor(torch.zeros, 2, 3)
        >>> x += 1
        >>> y = x * x
        >>> y = y.cuda().half()
        >>> y[0, 0] = 0
        >>> y = y.materialize()     # materialize the tensor
        >>> print(y)
        tensor([[0., 1., 1.],
                [1., 1., 1.]], device='cuda:0', dtype=torch.float16)

    Warnings:
        1. Cases that ``LazyTensor`` can't deal with.
        >>> x = LazyTensor(torch.ones, 2, 3)
        >>> x[0, 0] = -x[0, 0]    # this will cause infinite recursion
        >>> y = x.clone()
        >>> x.add_(1) # modifying origin tensor after cloning leads to wrong materialization
        >>> z = x.tolist()
        >>> x.zeros_() # modifying origin tensor after cloning tolist is not allowed
        >>> nn.utils.weight_norm(self.conv, name="weight", dim=2) # applying weight norm on a lazy tensor is not allowed


        2. Cases that ``LazyTensor`` becomes eager (early materialization).
        >>> b = a[:, 2:]  # get a slice of a lazy tensor triggers early materialization
        >>> chunks = a.split(3)  # this also triggers early materialization
        >>> x.data = torch.rand(2, 3) # directly setting data of a lazy tensor triggers early materialization

    """

    _repr = True
    _meta_data: Optional[MetaTensor] = None    # shape, dtype, device
    _pre_op_fn: Callable[['LazyTensor'], None] = lambda *args: None

    @staticmethod
    def __new__(cls, func, *args, meta_data=None, concrete_data=None, **kwargs):
        if concrete_data is not None:
            # some ops don't support meta backend and should have concrete data
            elem = concrete_data
        else:
            if meta_data is None:
                device = kwargs.get('device', 'cpu')
                elem = func(*args, **{**kwargs, 'device': 'meta'})
                meta_data = MetaTensor(elem, fake_device=device)
            elem = meta_data._tensor
        # As a meta tensor cannot be modified __class__ to torch.Tensor, we should use an empty real tensor here
        r = torch.Tensor._make_subclass(cls, _EMPTY_DATA, require_grad=elem.requires_grad)
        r._meta_data = meta_data
        return r

    def __init__(self, func, *args, meta_data=None, concrete_data=None, **kwargs):
        self._factory_method = (func, args, kwargs)    # (func, args, kwargs)
        self._op_buffer = []    # (func, args, kwargs, replace)
        self._materialized_data: Optional[torch.Tensor] = concrete_data    # materialized data

    def materialize(self) -> torch.Tensor:
        """Materialize the ``LazyTensor`` to ``torch.Tensor`` by modifying __class__ (inplace).

        Returns:
            torch.Tensor: The materialized tensor (self).
        """
        target = self._materialize_data()
        self.clean()
        cls_to_become = nn.Parameter if isinstance(self, nn.Parameter) else torch.Tensor
        self.__class__ = cls_to_become
        self.data = target
        return self

    def distribute(self, layout: Layout) -> torch.Tensor:
        """Distribute the ``LazyTensor`` to ``torch.Tensor`` by modifying __class__ (inplace), according to the layout.

        Args:
            layout (Layout): Distribution layout.

        Returns:
            torch.Tensor: The distributed tensor (self).
        """
        target = self._materialize_data()
        self.clean()
        local_tensor = DTensor(target, layout).local_tensor
        cls_to_become = nn.Parameter if isinstance(self, nn.Parameter) else torch.Tensor
        self.__class__ = cls_to_become
        self.data = local_tensor
        return self

    def clean(self) -> None:
        """Clean all stored operations and meta data, which prevents memory leaking. This should be called after all tensors are materialized.
        """
        self._factory_method = None
        self._op_buffer = None
        self._meta_data = None
        self._materialized_data = None

    @staticmethod
    def _replace_with_materialized(x):
        if isinstance(x, LazyTensor):
            return x._materialize_data()
        return x

    def _materialize_data(self) -> torch.Tensor:
        # self._materialized_data should be generated after the first call of this function
        if self._materialized_data is None:
            # apply factory method
            func, args, kwargs = self._factory_method

            # apply cached sequence
            self._pre_op_fn()

            try:
                init_val = func(*tree_map(self._replace_with_materialized, args),
                                **tree_map(self._replace_with_materialized, kwargs))
            except TypeError as e:
                print(f'init fn: {func.__name__}')
                raise e

            self._materialized_data = self._rerun_ops(init_val)
        return self._materialized_data

    def _rerun_ops(self, target=None) -> torch.Tensor:
        """Do lazy execution by rerunning all (stored) related operations.

        Args:
            target (torc.Tensor, optional): Intial value of the target tensor (self). Defaults to None.
        """

        def replace(x):
            if x is self:
                return target
            elif isinstance(x, LazyTensor):
                return x._materialize_data()
            return x

        packed = None

        for (func, args, kwargs) in self._op_buffer:
            if func == torch.Tensor.requires_grad_:
                packed = func, args, kwargs    # requires grad should be set at last
            else:
                self._pre_op_fn()
                o = func(*tree_map(replace, args), **tree_map(replace, kwargs))
                target = o if isinstance(o, torch.Tensor) else target    # if func returns non-Tensor, discard the value

        # super-dainiu: set requires_grad after all inplace-ops are done
        if packed is not None:
            func, args, kwargs = packed
            func(*tree_map(replace, args), **tree_map(replace, kwargs))

        return target

    # cache everything with __torch_function__

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func.__name__ in _EARLY_MATERIALIZED_OPS:
            # These OPs cannot be lazy and related tensors should be early materialized
            tree_map(cls._replace_with_materialized, args)
            tree_map(cls._replace_with_materialized, kwargs)
        is_inplace: bool = (func.__name__.endswith('_') and not (func.__name__.endswith('__'))
                            or func.__name__ == "__setitem__")

        is_change_meta_op: bool = func.__name__ in _CHANGE_META_OPS

        if isinstance(func, torch._C.ScriptMethod):
            # FIXME(ver217): torch script functions are not verified

            target = None

            def unwrap(x):
                if isinstance(x, LazyTensor):
                    return x._meta_data
                return x

            target: LazyTensor = args[0].clone()
            target._op_buffer.append((func, args, kwargs))
            target._meta_data = getattr(target._meta_data, func.name)(*tree_map(unwrap, args[1:]),
                                                                      **tree_map(unwrap, kwargs))
            return target
        else:

            meta_to_lazy = {}

            def unwrap(x):
                if isinstance(x, LazyTensor):
                    if x._materialized_data is not None:
                        # for early materialized tensor, use its materialized data directly
                        return x._materialized_data if is_change_meta_op else x._materialized_data.data
                    t = x if is_inplace else x.clone()
                    t._op_buffer.append((func, args, kwargs))
                    meta = x._meta_data if is_change_meta_op else x._meta_data.data
                    meta_to_lazy[meta] = t
                    return meta
                return x

            def wrap(y, i=None):
                if isinstance(y, MetaTensor):
                    if y in meta_to_lazy:
                        # inplace op, just return origin lazy tensor
                        return meta_to_lazy[y]
                    else:
                        # out of place op, create new lazy tensor
                        fn = lambda *a, **kw: func(*a, **kw) if i is None else func(*a, **kw)[i]
                        lazy_y = LazyTensor(fn, *args, meta_data=y, **kwargs)
                        return lazy_y
                elif type(y) is Tensor:
                    # for early materialized tensor
                    return LazyTensor(lambda: None, concrete_data=y)
                return y

            cls._pre_op_fn()
            o = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
            if isinstance(o, (tuple, list)):
                return type(o)(wrap(y, i=i) for i, y in enumerate(o))
            return wrap(o)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        pass    # skip

    def clone(self) -> "LazyTensor":

        def factory_fn():
            return self.materialize().clone()

        target = LazyTensor(factory_fn, meta_data=self._meta_data)

        return target

    def detach(self) -> Tensor:
        return self

    @property
    def data(self):
        return self

    @data.setter
    def data(self, other: 'LazyTensor'):
        """This is sightly different from oringinal `data` setter.

        E.g.:
            >>> a = torch.randn(3, 3) # a is a Tensor
            >>> b = torch.rand(2, 2)
            >>> a.data = b
            >>> b.add_(1)   # this will affect a
            >>> x = torch.randn(3, 3) # x is a LazyTensor
            >>> y = torch.rand(2, 2) # y is a LazyTensor
            >>> x.data = y
            >>> y.add_(1)   # this will not affect x

        """
        if other is self:
            return

        self._op_buffer.append(other._factory_method)

        def replace(x):
            if x is other:
                return self
            return x

        for func, args, kwargs in other._op_buffer:
            self._op_buffer.append((func, tree_map(replace, args), tree_map(replace, kwargs)))

    def tolist(self) -> list:
        # Though self.__class__ is modified to torch.Tensor, in C++ side, it is still a subclass of torch.Tensor
        # And subclass of torch.Tensor does not have tolist() method
        t = self._materialize_data()
        return t.tolist()

    def __hash__(self):
        return id(self)


class LazyInitContext:
    """Context manager for lazy initialization. Enables initializing the model without allocating real memory.

    Usage:
        1. The model is initialized, but no real memory is allocated.
        >>> ctx = LazyInitContext()
        >>> with ctx:
        >>>     model = MyModel().cuda()

        2. The model is initialized with ``MetaTensor`` as weights, but still no real memory is allocated.
        >>> with ctx.traceable(model):
        >>>     gm = symbolic_trace(model, meta_args=meta_args)
        >>> # Solve the execution strategy and apply the strategy to the model
        >>> strategy = StrategyAndSpec()

        3. The model is initialized with ``torch.Tensor`` as weights, and real memory is allocated. (single device)
        >>> model = ctx.materialize(model)

        3. The model is initialized with sharded ``torch.Tensor`` as weights, and real memory is allocated. (distributed scenario)
        >>> model = apply_strategy_to_all_params(model, strategy)
        >>> model = ctx.distribute(model)

    Warnings:
        This API is still experimental and further modifications can be made to it.
        For example:
            1. Quantization strategies can be applied before allocating real memory.
            2. Lazy initialization seems slower than normal initialization.
    """
    _replaced: bool = False

    def __init__(self, tensor_cls: Union[_MyTensor, LazyTensor] = LazyTensor):
        self.overrides = {}
        self.tensor_cls = tensor_cls

    def __enter__(self):
        if LazyInitContext._replaced:
            raise RuntimeError(f'LazyInitContext is not reentrant')
        LazyInitContext._replaced = True

        def wrap_factory_method(target):
            # factory functions (eg. torch.empty())
            def wrapper(*args, **kwargs):
                return self.tensor_cls(target, *args, **kwargs)

            return wrapper, target

        def wrap_factory_like_method(orig_target, target):
            # factory_like functions (eg. torch.empty_like())
            def wrapper(*args, **kwargs):
                orig_t = args[0]
                return self.tensor_cls(orig_target, *args[1:], device=orig_t.device, dtype=orig_t.dtype, **kwargs)

            return wrapper, target

        def wrap_legacy_constructor(target, dtype):
            # legacy constructor (e.g. torch.LongTensor())
            def wrapper(*args, **kwargs):
                if len(args) == 1 and isinstance(args[0], torch.Tensor):
                    # (Tensor other)
                    return args[0]
                elif len(args) == 1:
                    # (object data, *, torch.device device)
                    kwargs = {**kwargs, 'dtype': dtype}
                    replaced, orig = self.overrides['tensor']
                    return replaced(*args, **kwargs)
                elif _is_int_tuple(args):
                    # (tuple of ints size, *, torch.device device)
                    kwargs = {**kwargs, 'dtype': dtype}
                    replaced, orig = self.overrides['empty']
                    return replaced(*args, **kwargs)
                else:
                    raise TypeError(
                        f'new() received an invalid combination of arguments - got {tuple(type(x) for x in args)}, but expected one of:\n * (Tensor other)\n * (tuple of ints size, *, torch.device device)\n * (object data, *, torch.device device)'
                    )

            return wrapper, target

        def wrap_no_meta_factory(target):
            # factory functions which don't support meta tensor backend
            def wrapper(*args, **kwargs):
                tensor = target(*args, **kwargs)
                return self.tensor_cls(lambda: None, concrete_data=tensor)

            return wrapper, target

        self.overrides = {
            target: wrap_factory_method(getattr(torch, target))
            for target in _NORMAL_FACTORY
            if callable(getattr(torch, target, None))
        }

        self.overrides.update({
            target + '_like': wrap_factory_like_method(getattr(torch, target), getattr(torch, target + '_like'))
            for target in _NORMAL_FACTORY
            if callable(getattr(torch, target + '_like', None))
        })

        self.overrides.update({
            target: wrap_legacy_constructor(getattr(torch, target), dtype)
            for target, dtype in _LEGACY_TENSOR_CONSTRUCTOR.items()
            if callable(getattr(torch, target, None))
        })

        self.overrides.update({
            target: wrap_no_meta_factory(getattr(torch, target))
            for target in _NO_META_FACTORY
            if callable(getattr(torch, target, None))
        })

        for name, (wrapper, orig) in self.overrides.items():
            setattr(torch, name, wrapper)

    def __exit__(self, exc_type, exc_val, exc_tb):
        LazyInitContext._replaced = False
        for name, (wrapper, orig) in self.overrides.items():
            setattr(torch, name, orig)

    @staticmethod
    def materialize(module: torch.nn.Module, verbose: bool = False) -> nn.Module:
        """Initialize all ``nn.Parameter`` from ``LazyTensor``. This function will modify the module in-place.

        Args:
            module (torch.nn.Module): Target ``nn.Module``
            verbose (bool): Whether to print lazy initialization rate. Defaults to False.
        """
        if verbose:
            # verbose info
            param_cnt = 0
            param_lazy_cnt = 0
            buf_cnt = 0
            buf_lazy_cnt = 0
            total_numel = 0
            non_lazy_numel = 0

        for name, p in module.named_parameters():
            param_cnt += 1
            total_numel += p.numel()
            if getattr(p, '_materialized_data', False) is None:
                # if no _materialized_data attr, the tensor is not lazy
                param_lazy_cnt += 1
            else:
                non_lazy_numel += p.numel()
            if isinstance(p, LazyTensor):
                p.materialize()

        for name, buf in module.named_buffers():
            buf_cnt += 1
            total_numel += buf.numel()
            if getattr(buf, "_materialized_data", False) is None:
                # if no _materialized_data attr, the tensor is not lazy
                buf_lazy_cnt += 1
            else:
                non_lazy_numel += buf.numel()
            if isinstance(buf, LazyTensor):
                buf.materialize()

        if verbose:
            non_lazy_numel_ratio = non_lazy_numel / total_numel * 100 if non_lazy_numel != 0 else 0
            print(f'Param lazy rate: {param_lazy_cnt}/{param_cnt}')
            print(f'Buffer lazy rate: {buf_lazy_cnt}/{buf_cnt}')
            print(f'Non lazy numel: {non_lazy_numel} ({non_lazy_numel/1024**2:.3f} M), ratio: {non_lazy_numel_ratio}%')

        return module


def _is_int_tuple(args) -> bool:
    if not isinstance(args, tuple):
        return False
    for x in args:
        if not isinstance(x, int):
            return False
    return True
