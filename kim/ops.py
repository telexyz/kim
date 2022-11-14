from numbers import Number
from typing import Optional, List
from .autograd import NDArray, array_api
from .autograd import Tensor, TensorOp
from .autograd import TensorTuple, TensorTupleOp

import numpy as np
from kim import backend_ndarray as nd
from kim import init


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple: return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])

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
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return make_tuple(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


# - - - - - - - - - - - -


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
        if len(b.shape) > len(a.shape): a = a.reshape(b.shape)
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad

def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = np.float32(scalar)
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
        self.scalar = np.float32(scalar)

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return mul_scalar(out_grad, self.scalar),

def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)



class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = np.float32(scalar)

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
            out_grad * negate(power_scalar(b, -2) * a)
        )

def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = np.float32(scalar)

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
    def __init__(self, axes: Optional[tuple] = None, keepdims=False):
        self.axes = axes	
        self.keepdims = keepdims

    def compute(self, a):
        return a.sum(self.axes, keepdims=self.keepdims)

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

        if len(out_grad.shape) == len(a.shape):
            x = out_grad
        else:
            x = reshape(out_grad, new_shape)

        return broadcast_to(x, a.shape),

def summation(a, axes=None, keepdims=False):
    return Summation(axes, keepdims=keepdims)(a)


class MatMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        ### numpy_backend
        if array_api == np: return a @ b

        ### ndarray_backend
        # 2D @ 2D
        if a.ndim == 2 and b.ndim == 2: return a @ b
        
        # 2D @ 3D
        if a.ndim == 2 and b.ndim == 3:
            assert a.shape[1] == b.shape[1], "Matrix sizes not matched"
            c = np.zeros((b.shape[0], a.shape[0], b.shape[2])).astype(a.dtype)
            for i in range(b.shape[0]):
                _b = b[i,:,:].compact().reshape((b.shape[1], b.shape[2]))
                c[i] = (a @ _b).numpy()
            return nd.array(c, device=a.device)
        
        # 3D @ 2D
        if a.ndim == 3 and b.ndim == 2:
            assert a.shape[2] == b.shape[0], "Matrix sizes not matched"
            c = np.zeros((a.shape[0], a.shape[1], b.shape[1])).astype(a.dtype)
            for i in range(a.shape[0]):
                _a = a[i,:,:].compact().reshape((a.shape[1], a.shape[2]))
                c[i] = (_a @ b).numpy()
            return nd.array(c, device=a.device)

        # 3D @ 3D
        # https://www.geeksforgeeks.org/numpy-3d-matrix-multiplication
        assert a.ndim == 3 and b.ndim == 3, "Only support 2D @ 2D, 3D @ 2D, and 3D @ 3D"
        assert a.shape[2] == b.shape[1], "Matrix sizes not matched"
        assert a.shape[0] == b.shape[0], "Batch need to be same size"
        c = np.zeros((a.shape[0], a.shape[1], b.shape[2])).astype(a.dtype)
        for i in range(a.shape[0]):
            _a = a[i,:,:].compact().reshape((a.shape[1], a.shape[2]))
            _b = b[i,:,:].compact().reshape((b.shape[1], b.shape[2]))
            c[i] = (_a @ _b).numpy()
        return nd.array(c, device=a.device)


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
        return array_api.log(a)

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
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        a = node.inputs[0].realize_cached_data()
        relu_a = Tensor(a > 0)
        return Tensor(out_grad * relu_a),

def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        Z_max = array_api.max(Z, axis=self.axes)
        Z_max_reshape = Z_max.reshape(self.new_shape(Z.shape))
        Z_max_broadcast = array_api.broadcast_to(Z_max_reshape, Z.shape)
        ZZ = Z - Z_max_broadcast
        return array_api.log(array_api.exp(ZZ).sum(self.axes)) + Z_max

    def gradient(self, out_grad, node):
        a = node.inputs[0].realize_cached_data()
        # print(">>>", array_api, array_api.max, type(a))
        m = a.max(axis=self.axes, keepdims=True)
        # print(">>>", a.shape, m.shape)
        exp_a = exp(Tensor(a - array_api.broadcast_to(m, a.shape)))
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
        if self.axes is None:
            axes = range(len(shape))
        else:
            axes = self.axes
            if isinstance(axes, int): axes = [axes]

        new_shape = list(shape)
        for i in axes: new_shape[i] = 1
        return tuple(new_shape)


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)



class Tanh(TensorOp):
    def compute(self, a):
        return array_api.tanh(a)

    def gradient(self, out_grad, node):
        '''adjoin w.r.t x = out_grad * derivate(f(x))
        d_tanh(x) = 1 - tanh(x)^2
        d_x = out_grad * (1 - tanh(x)^2)
        '''
        x = node.inputs[0]
        y = 1 + negate(power_scalar(tanh(x), 2))
        return (out_grad * y,)

def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        # https://www.geeksforgeeks.org/python-pytorch-stack-method
        ### BEGIN YOUR SOLUTION
        # print(">>> args:", len(args), args[0].shape, self.axis)
        shape = list(args[0].shape)
        # print("---", shape)
        shape.insert(self.axis, len(args))
        # print("+++", shape)
        idxs = [ slice(0,shape[i],1) for i in range(len(shape)) ]
        # print("@@@", idxs)
        out = NDArray.make(shape, device=args[0].device)
        # print(">>> out:", out.shape)
        for i in range(len(args)):
            assert args[0].shape == args[i].shape, "stacked tensors must be same shape"
            idxs[self.axis] = slice(i,i+1,1)
            # print(">>> slice:", idxs)
            out.__setitem__(tuple(idxs), args[i])
        return out
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # print(">>> node:", type(node), len(node.inputs))
        a = split(out_grad, self.axis).realize_cached_data()
        return make_tuple(*[Tensor(x) for x in a]),
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))

import copy
class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        print(">>> A:", A.shape, self.axis)
        shape = list(A.shape)
        idxs = [ slice(0,shape[i],1) for i in range(len(shape)) ]
        b_idxs = copy.deepcopy(idxs)
        del b_idxs[self.axis] # xóa phần tử thứ self.axis của b_idxs
        del shape[self.axis] # xóa phần tử thứ self.axis của shape
        print(">>> shape:", shape)
        out = []
        for i in range(A.shape[self.axis]):
            idxs[self.axis] = slice(i,i+1,1)
            a = A.__getitem__(tuple(idxs))
            b = NDArray.make(shape, device=A.device)
            b.__setitem__(tuple(b_idxs), a)
            out.append(b)
        return tuple(out)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # print(">>> node:", len(node), type(node))
        # print(">>> node.inputs:", len(node.inputs), type(node.inputs))
        ### BEGIN YOUR SOLUTION
        # A = node.inputs[0]
        # print("\n>>> out_grad:", out_grad.shape, out_grad, type(out_grad))
        # print(">>> A:", A.shape, type(A))
        # print(">>> axis:", self.axis)
        # n = A.shape[self.axis]
        # return stack([out_grad for i in range(n)], self.axis)
        raise NotImplementedError()
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return a.flip(self.axes)

    def gradient(self, out_grad, node):
        return flip(out_grad, self.axes),


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        idxs = [slice(0, a.shape[i], 1) for i in range(len(a.shape))]
        for axis in self.axes:
            new_shape[axis] *= (self.dilation + 1)
            # 1 cho phần tử gốc và self.dilation cho 0 padding
            idxs[axis] = slice(0, new_shape[axis], self.dilation + 1)
        out = a.device.zeros(*new_shape)
        out.__setitem__(tuple(idxs), a.compact())
        return out.compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation),
        ### END YOUR SOLUTION

def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        idxs = [slice(0, a.shape[i], 1) for i in range(len(a.shape))]
        for axis in self.axes:
            new_shape[axis] //= (self.dilation + 1)
            idxs[axis] = slice(0, a.shape[axis], self.dilation + 1)

        out = a.compact().__getitem__(tuple(idxs))
        return out.compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, onde):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
