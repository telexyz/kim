from . import ops
from .ops import *
from .autograd import Tensor
from .backend_selection import *

from . import init
from . import data
from . import nn
from . import optim

# Utils to switch between ndarray_backend and numpy_backend
import numpy as np
def as_numpy(x):
    if isinstance(x, np.ndarray): return x
    if isinstance(x, np.float32): return x
    if isinstance(x, np.int64): return x
    return x.numpy()
