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
            # bias_init = init.kaiming_uniform(1, out_features, dtype=dtype)
            # self.bias = Parameter(bias_init)
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        mm = X @ self.weight
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

        x = ops.add(logsum, ops.negate(logits_y))
        # workaround to convert logits.shape[0] to logits.dtype
        n = np.array([logits.shape[0]], dtype=logits.dtype)[0]
        return ops.summation(x) / n


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
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(1, 1, dtype=dtype))
        self.bias = Parameter(init.zeros(1, 1, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[1] == self.dim
        n = x.shape[0]
        # print(">>>", x.shape, n, self.dim)
        mean = ops.summation(x, axes=1)
        mean = ops.divide_scalar(mean, self.dim)
        mean = ops.reshape(mean, (n, 1))
        mean = ops.broadcast_to(mean, x.shape)

        x2 = ops.power_scalar(x, 2)
        mean_x2 = ops.summation(x2, axes=1)
        mean_x2 = ops.divide_scalar(mean_x2, self.dim)
        mean_x2 = ops.reshape(mean_x2, (n, 1))
        mean_x2 = ops.broadcast_to(mean_x2, x.shape)

        var = mean_x2 - ops.power_scalar(mean, 2)
        var = ops.add_scalar(var, self.eps)
        var = ops.power_scalar(var, 1/2)

        x = x + (-1)*mean
        x = x / var

        w = ops.broadcast_to(self.weight, x.shape)
        b = ops.broadcast_to(self.bias, x.shape)
        y = ops.multiply(w, x) + b
        return y