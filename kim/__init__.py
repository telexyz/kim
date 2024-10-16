from . import ops
from .ops import *
from .autograd import Tensor
from .backend_selection import *

from . import init
from . import data
from . import nn
from . import optim
from . import utils
from . import timelog
from .utils import as_numpy, prod

KIM_FUSE = os.environ.get("KIM_FUSE", False)
