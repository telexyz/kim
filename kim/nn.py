"""The module.
"""
from typing import List, Callable, Any
from kim.autograd import Tensor
from kim import ops
import kim.init as init
import numpy as np

class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""

def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        weight_init = init.kaiming_uniform(in_features, out_features, dtype=dtype)
        self.weight = Parameter(weight_init)
        if bias is True:
            bias_init = init.kaiming_uniform(out_features, 1, dtype=dtype)
            self.bias = Parameter(ops.transpose(bias_init))
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        mm = ops.matmul(X, self.weight)
        if self.bias is None:
            return mm
        else:
            return mm + ops.broadcast_to(self.bias, mm.shape)


class Flatten(Module):
    # Takes in a tensor of shape (B,X_0,X_1,...), and flattens all non-batch dimensions so that the output is of shape (B, X_0 * X_1 * ...)
    def forward(self, X):
        m = 1
        for i in range(len(X.shape)-1):
            m = m * X.shape[i + 1]
        new_shape = (X.shape[0], m)
        return ops.reshape(X, new_shape)


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for m in self.modules: x = m(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        y_one_hot = init.one_hot(logits.shape[1], y, dtype=logits.dtype)
        logsum = ops.logsumexp(logits, axes=(1,))
        logits_y = ops.summation(logits * y_one_hot, axes=(1,))
        loss = logsum - logits_y
        return ops.summation(loss) / logits.shape[0]


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = 1 - p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mask = init.randb(x.shape[0], x.shape[1], p=self.p, dtype=x.dtype)
            return (x * mask) / self.p
        return x


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    # Given module F and input Tensor x, returning F(x) + x
    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(1, self.dim, dtype=dtype))
        self.bias = Parameter(init.zeros(1, self.dim, dtype=dtype))
        self.running_mean = init.zeros(self.dim, dtype=dtype)
        self.running_var = init.ones(self.dim, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        batch, dim = x.shape
        assert dim == self.dim

        if self.training:
            # only calculate mean, var and update running_mean, running_var in training
            mean = ops.summation(x, axes=0) / batch
            mean_full = mean.reshape((1, dim)).broadcast_to(x.shape)
            var = ops.summation((x - mean_full) ** 2, axes=0) / batch

            self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mean
            self.running_var  = (1-self.momentum)*self.running_var  + self.momentum*var
        else:
            # inference use running_mean and running_var estimated in training
            mean_full = self.running_mean.reshape((1, dim)).broadcast_to(x.shape)
            var = self.running_var

        var_full = var.reshape((1, dim)).broadcast_to(x.shape)
        norm = (x - mean_full) / ((var_full + self.eps) ** 0.5)
        
        w = self.weight.broadcast_to(x.shape)
        b = self.bias.broadcast_to(x.shape)
        return w * norm + b

'''
https://www.geeksforgeeks.org/expression-for-mean-and-variance-in-a-running-stream
https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
'''
class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(1, self.dim, dtype=dtype))
        self.bias = Parameter(init.zeros(1, self.dim, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        batch, dim = x.shape
        assert dim == self.dim

        mean = ops.summation(x, axes=1) / dim
        mean = mean.reshape((batch, 1)).broadcast_to(x.shape)

        var = ops.power_scalar(x - mean, 2)
        var = ops.summation(var, axes=1) / dim
        var = var.reshape((batch, 1)).broadcast_to(x.shape)

        norm = (x - mean) / ((var + self.eps) ** 0.5)
        w = ops.broadcast_to(self.weight, x.shape)
        b = ops.broadcast_to(self.bias, x.shape)
        return w*norm + b
