from numbers import Number
from typing import Optional, List
from .autograd import NDArray, array_api
from .autograd import Tensor, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Tensor:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


'''Lưu ý: các hàm tính gradient() của các TensorOp được định nghĩa dưới đây
dùng ngay chính các toán tử TensorOp để tạo ra một đồ thị tính toán ngược để có thể 
bắt đầu quy trình backward.
'''

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad

def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = numpy.float32(scalar)
        # self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad,

def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        a, b = node.inputs
        return multiply(out_grad, b), multiply(out_grad, a)

def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = numpy.float32(scalar)

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return mul_scalar(out_grad, self.scalar),

def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)



class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = numpy.float32(scalar)

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        # adjoin w.r.t a = out_grad * grad(a^scalar)
        #                = out_grad * scalar * a ^ (scalar - a)
        a = node.inputs[0]
        grad = power_scalar(a, self.scalar - 1)
        grad = mul_scalar(grad, self.scalar)
        return multiply(out_grad, grad),

def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""
    def compute(self, a, b):
        return a / b

    ''' Chain rule: gradient(a) = out_grad * gradient_a(f(a,b))
        gradient_a(a / b) = 1 / b
    https://www.cuemath.com/questions/find-the-derivative-of-1-x
    gradient_b(a / b):
    vì gradient(1/b) = gradient(b^-1)
    mà gradient(b^n) =  n * b^(n - 1)
    => gradient(1/b) = -1 * b^(-2) = -power(b, -2)
    => gradient_b(a / b) = -power(b, -2) * a
    '''
    def gradient(self, out_grad: Tensor, node: Tensor):
        a, b = node.inputs
        return (
            divide(out_grad, b), 
            multiply(out_grad, multiply(negate(power_scalar(b, -2)), a))
        )

def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = numpy.float32(scalar)

    def compute(self, a: NDArray):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return divide_scalar(out_grad, self.scalar),

def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes:
            return a.swapaxes(self.axes[0], self.axes[1])
        else:
        	n = len(a.shape)
        	return a.swapaxes(n-1, n-2)

    def gradient(self, out_grad, node):
        return transpose(out_grad, self.axes),

def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return a.reshape(self.shape)

    def gradient(self, out_grad, node):
        return reshape(out_grad, node.inputs[0].shape),

def reshape(a, shape):
	return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        n = len(a.shape)
        # print(">>>", a.shape, self.shape)
        if n < len(self.shape):
            shape = []
            k = 0
            for i in range(len(self.shape)):
                if n == 0 or k == n:
                    shape.insert(0, 1)
                else:
                    shape.append(a.shape[k])
                    k += 1
            # print(shape)
            a = a.reshape(tuple(shape))

        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        a = node.inputs[0]

        axes = ()
        n = len(a.shape)
        if n == len(self.shape):
            for i in range(len(self.shape)):
                if i >= n or self.shape[i] != a.shape[i]:
                    axes += (i,)
        else:
            k = 0
            for i in range(len(self.shape)):
                if n == 0 or self.shape[i] != a.shape[k]:
                    axes += (i,)
                else:
                    k += 1

        accum_grads = summation(out_grad, axes=axes)
        return reshape(accum_grads, a.shape),

def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes	

    def compute(self, a):
        return a.sum(self.axes)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        new_shape = out_grad.shape
        
        axes = self.axes
        if axes is None: axes = ()        
        # Trường hợp axes là int thì convert về tuple
        if not isinstance(axes, tuple): axes = (axes,)

        for i in range(len(axes)):
            idx = axes[i]
            new_shape = new_shape[0:idx] + (1,) + new_shape[idx:]        
        # Các thao tác trên chỉ để tính new_shape

        x = reshape(out_grad, new_shape)
        return broadcast_to(x, a.shape),

def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        if array_api == numpy: return a @ b
        if a.ndim == 2: return a @ b

        # batch matmul
        assert a.ndim == 3
        assert b.ndim == 3
        assert a.shape[0] == b.shape[0]
        assert a.shape[2] == b.shape[1]

        c = numpy.zeros((a.shape[0], a.shape[1], b.shape[2])).astype(a.dtype)
        for i in range(a.shape[0]):
            _a = a[i,:,:].compact().reshape((a.shape[1], a.shape[2]))
            _b = b[i,:,:].compact().reshape((b.shape[1], b.shape[2]))
            c[i] = (_a @ _b).numpy()
        return Tensor(c)


    def gradient(self, out_grad, node):
        a, b = node.inputs 

        b_transpose = transpose(b)
        l_grad = matmul(out_grad, b_transpose)
        # chuẩn hóa shape l_grad
        n = len(l_grad.shape) - len(a.shape)
        if n > 0:
            axes = ()
            for i in range(n): 
                axes += (i,)
            l_grad = summation(l_grad, axes=axes)


        a_transpose = transpose(a)
        r_grad = matmul(a_transpose, out_grad)
        # chuẩn hóa shape r_grad
        n = len(r_grad.shape) - len(b.shape)
        if n > 0:
            axes = ()
            for i in range(n): axes += (i,)
            r_grad = summation(r_grad, axes=axes)

        return (l_grad, r_grad)

def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return negate(out_grad),

def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        if isinstance(a, numpy.ndarray):
            return numpy.log(a)
        else:
            return a.log()

    def gradient(self, out_grad, node):
        return divide(out_grad, node.inputs[0]),

def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        # đạo hàm e^x là e^x
        return multiply(out_grad, exp(node.inputs[0])),

def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        if isinstance(a, numpy.ndarray):
            return numpy.maximum(a, 0)
        else:
            return a.maximum(0)

    def gradient(self, out_grad, node):
        a = node.inputs[0].numpy()
        relu_a = (a > 0)
        return Tensor(out_grad.numpy() * relu_a),

def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        if array_api is numpy:
            Z_max = array_api.max(Z, axis=self.axes)
            Z_max_reshape = array_api.reshape(Z_max, self.new_shape(Z.shape))
            Z_max_broadcast = array_api.broadcast_to(Z_max_reshape, Z.shape)
            ZZ = Z - Z_max_broadcast
            exp_ZZ = array_api.exp(ZZ)
            sum_exp_ZZ = array_api.sum(exp_ZZ, self.axes)
            return array_api.log(sum_exp_ZZ) + Z_max
        else:
            assert len(self.axes) == 1
            Z_max = Z.max(self.axes[0])

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        m = array_api.max(a.numpy(), axis=self.axes, keepdims=True)
        exp_a = exp(a - m)
        # 
        new_shape = self.new_shape(a.shape)
        sum_exp_a = summation(exp_a, self.axes)
        sum_exp_a = reshape(sum_exp_a, new_shape)
        sum_exp_a = broadcast_to(sum_exp_a, a.shape)
        # 
        normalize = divide(exp_a, sum_exp_a)
        y = reshape(out_grad, new_shape)
        y = broadcast_to(y, a.shape)
        return (y * normalize,)

    def new_shape(self, shape):
        new_shape = list(shape)
        if self.axes:
            axes = self.axes
        else:
            axes = range(len(shape))
        for i in axes: new_shape[i] = 1
        return tuple(new_shape)

def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
