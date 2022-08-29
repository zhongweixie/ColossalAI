try:
    from ._meta_registrations import *
except:
    print(f'_meta_registrations seems to be incompatible with PyTorch {torch.__version__}.')
from .meta_tensor import *
from .registry import *
from .profiler_function import *
from .profiler_module import *
from .profiler import *
