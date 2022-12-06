from numbers import Number
from typing import Optional, List, Tuple
from .autograd import NDArray, array_api
from .autograd import Tensor, TensorOp
from .autograd import TensorTuple, TensorTupleOp

import numpy as np
from kim import backend_ndarray as nd
from kim import init

# numpy backend vs other backend united interfaces
def compact(a):
    if array_api == np: return a
    else: return a.compact()

def make(ndarray, shape, array):
    if ndarray == np.ndarray: return np.empty(shape)
    else: return ndarray.make(shape, device=array.device)

# - - - -

class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple: return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        n_inputs = len(node.inputs)
        assert n_inputs == len(out_grad)
        # trả lại gradient thông qua ops.tuple_get_item
        assert not isinstance(out_grad.op, MakeTensorTuple), "Cần thêm code https://github.com/dlsyscourse/hw4/blob/main/python/needle/ops.py#L31"
        return tuple([tuple_get_item(out_grad, index) for index in range(n_inputs)])

def make_tensor_tuple(*args):
    return MakeTensorTuple()(*args)

class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        n_inputs = len(node.inputs[0])
        in_grads = [ init.zeros_like(out_grad) ] * (n_inputs - 1)
        in_grads.insert(self.index, out_grad)
        return make_tensor_tuple(*in_grads),

def tuple_get_item(a, index):
    return TupleGetItem(index)(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, arrays: Tuple[NDArray]) -> NDArray:
        # Vì compute luôn sử dụng dữ liệu từ .cached_data nên arrays ở đây phải là tupple của NDArray
        assert isinstance(arrays, tuple)
        array0 = arrays[0]
        assert isinstance(array0, NDArray)
        shape = list(array0.shape)
        shape.insert(self.axis, len(arrays)) # thêm 1 chiều không gian nữa
        idxs = [ slice(0, shape[i], 1) for i in range(len(shape)) ]
        out = make(NDArray, shape, array0)
        for i, arrayi in enumerate(arrays):
            assert array0.shape == arrayi.shape, "stacked arrays must be same shape"
            idxs[self.axis] = slice(i, i+1, 1) # gán dữ liệu vào từng lát cát của chiều self.axis
            out.__setitem__(tuple(idxs), arrayi)
        return out
        # https://www.geeksforgeeks.org/python-pytorch-stack-method

    def gradient(self, out_grad, node):
        return split(out_grad, self.axis),

def stack(args, axis):
    tensor_tuple = make_tensor_tuple(*args)
    return Stack(axis)(tensor_tuple)


class Split(TensorTupleOp):
    def __init__(self, axis: int, chunks=None):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis
        self.chunks = chunks

    def compute(self, A: NDArray):
        shape = list(A.shape)
        idxs = [ slice(0,shape[i],1) for i in range(len(shape)) ]

        if self.chunks is None:
            del shape[self.axis] # xóa phần tử thứ self.axis của shape
            chunks = A.shape[self.axis]
            offset = 1
        else:
            chunks = self.chunks
            offset = A.shape[self.axis] // chunks
            shape[self.axis] = offset

        out = []
        for i in range(chunks):
            start = i * offset
            idxs[self.axis] = slice(start, start + offset, 1)
            a = A.__getitem__(tuple(idxs))
            out.append(compact(a).reshape(shape))
        # 
        return tuple(out)


    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        assert isinstance(out_grad, TensorTuple)
        return Stack(self.axis)(out_grad).reshape(input_shape),


def split(a, axis, chunks=None):
    return Split(axis, chunks=chunks)(a)


# - - - - - - - - - - - -


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
        self.new_shape = None

    def compute(self, a):
        n = len(a.shape)
        if n == len(self.shape):
            self.new_shape = a.shape
        else:
            new_shape = []
            k = 0
            for i in range(len(self.shape)):
                if n == 0 or k == n:
                    new_shape.insert(0, 1)
                else:
                    new_shape.append(a.shape[k])
                    k += 1

            self.new_shape = tuple(new_shape)
            a = a.reshape(self.new_shape)

        return array_api.broadcast_to(a, self.shape)


    def gradient(self, out_grad, node):
        a = node.inputs[0]
        axes = ()
        for i, x in enumerate(self.new_shape):
            if x == 1: axes += (i,)
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
        
        if len(out_grad.shape) != len(a.shape) and self.axes is not None:
            if isinstance(self.axes, int): axes = (self.axes,)
            else: axes = tuple(self.axes)

            new_shape = out_grad.shape
            for idx in sorted(axes): 
                new_shape = new_shape[0:idx] + (1,) + new_shape[idx:]
            out_grad = reshape(out_grad, new_shape)
        #
        return broadcast_to(out_grad, a.shape),

def summation(a, axes=None, keepdims=False):
    return Summation(axes, keepdims=keepdims)(a)


class MatMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        ### numpy_backend
        if array_api == np: return a @ b

        ### ndarray_backend
        assert a.shape[-1] == b.shape[-2], "MatMul: Sizes not matched %s @ %s" % (a.shape, b.shape)

        # 2D @ 2D
        if a.ndim == 2 and b.ndim == 2: return a @ b
        
        # 2D @ nD
        if a.ndim == 2:
            if b.ndim == 3:
                c = b.permute((1,2,0)).compact().reshape((b.shape[1], b.shape[2]*b.shape[0]))
                return (a @ c).reshape((a.shape[0], b.shape[2], b.shape[0])).permute((2,0,1))

            assert b.ndim == 4, "MatMul: Only support 2D @ 3,4D MatMul"
            c = b.permute((2,3,0,1)).compact().reshape((b.shape[2], b.shape[3]*b.shape[0]*b.shape[1]))
            return (a @ c).reshape((a.shape[0], b.shape[3], b.shape[0], b.shape[1])).permute((3,0,1,2))

        # nD @ 2D
        if b.ndim == 2:
            assert a.ndim >= 3, "MatMul: a.ndim must >= 3 for nD @ 2D"
            c = a.reshape((-1, a.shape[-1]))
            shape = list(a.shape)
            shape[-1] = b.shape[1]
            return (c @ b).reshape(tuple(shape))

        # 3D @ 3D
        # https://www.geeksforgeeks.org/numpy-3d-matrix-multiplication
        if a.ndim == 3 and b.ndim == 3:
            assert a.shape[0] == b.shape[0], "MatMul: Batch need to be same size"
            c = NDArray.make((a.shape[0], a.shape[-2], b.shape[-1]), device=a.device)
            for i in range(a.shape[0]):
                _a = a[i,:,:].compact().reshape((a.shape[-2], a.shape[-1]))
                _b = b[i,:,:].compact().reshape((b.shape[-2], b.shape[-1]))
                c[i,:,:] = (_a @ _b).reshape((1, a.shape[-2], b.shape[-1]))
            return c

        # >>> (3, 2, 1) (3, 3, 1, 2)
        if b.ndim == 4 and a.ndim == 3 and b.shape[1] == a.shape[0]:
            c = NDArray.make((b.shape[0], a.shape[-3], a.shape[-2], b.shape[-1]), device=a.device)
            for i in range(a.shape[0]):
                _a = a[i,:,:].compact().reshape((a.shape[-2], a.shape[-1]))
                for j in range(b.shape[0]):
                    _b = b[j,i,:,:].compact().reshape((b.shape[-2], b.shape[-1]))
                    c[j,i,:,:] = (_a @ _b).reshape((1, 1, a.shape[-2], b.shape[-1]))
            return c

        # >>> (3, 3, 2, 2) (3, 3, 2, 1)
        if b.ndim == 4 and a.ndim == 4 and b.shape[0] == a.shape[0] and b.shape[1] == a.shape[1]:
            c = NDArray.make((a.shape[-4], a.shape[-3], a.shape[-2], b.shape[-1]), device=a.device)
            for i in range(a.shape[0]):
                for j in range(b.shape[1]):
                    _a = a[i,j,:,:].compact().reshape((a.shape[-2], a.shape[-1]))
                    _b = b[i,j,:,:].compact().reshape((b.shape[-2], b.shape[-1]))
                    c[i,j,:,:] = (_a @ _b).reshape((1, 1, a.shape[-2], b.shape[-1]))
            return c

        print(">>>", a.shape, b.shape)
        assert False, "MatMul: Only support 2D @ 2D, nD @ 2D, 2D @ 3:4D, 3D @ 3D and selected 4D @ 4D"


    def gradient(self, out_grad, node):
        a, b = node.inputs 

        l_grad = matmul(out_grad, transpose(b))
        n = len(l_grad.shape) - len(a.shape)
        if n > 0: l_grad = summation(l_grad, axes=tuple(i for i in range(n)))

        r_grad = matmul(transpose(a), out_grad)
        n = len(r_grad.shape) - len(b.shape)
        if n > 0: r_grad = summation(r_grad, axes=tuple(i for i in range(n)))

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
        relu_a = Tensor(a > 0, device=out_grad.device)
        return out_grad * relu_a,

def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        Z_max = array_api.max(Z, axis=self.axes)
        ZZ = Z_max.reshape(self.new_shape(Z.shape))
        ZZ = array_api.broadcast_to(ZZ, Z.shape)
        ZZ = array_api.exp(Z - ZZ).sum(self.axes)
        return array_api.log(ZZ) + Z_max

    def gradient(self, out_grad, node):
        a = node.inputs[0].realize_cached_data()
        m = a.max(axis=self.axes, keepdims=True)
        exp_a = exp(Tensor(a - array_api.broadcast_to(m, a.shape), device=out_grad.device))
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


class SoftMax(TensorOp):
    ''' https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
    exp(e_i)/sum(exp(e_k)) k=0..n
    '''
    def compute(self, z: NDArray) -> NDArray:
        z = array_api.exp(z - z.max(axis=-1, keepdims=True))
        return z / z.sum(axis=-1, keepdims=True)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        # z = node.inputs[0]
        # s = softmax(z)
        # return out_grad * ...
        raise NotImplementedError()

def softmax(z):
    return SoftMax()(z)


class Tanh(TensorOp):
    def compute(self, a):
        return array_api.tanh(a)

    def gradient(self, out_grad, node):
        '''adjoin w.r.t x = out_grad * derivate(f(x))
        d_tanh(x) = 1 - tanh(x)^2
        d_x = out_grad * (1 - tanh(x)^2)
        '''
        x = node.inputs[0]
        y = 1 - (tanh(x) ** 2)
        return (out_grad * y,)

def tanh(a):
    return Tanh()(a)



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
        new_shape = list(a.shape)
        idxs = [slice(0, a.shape[i], 1) for i in range(len(a.shape))]
        for axis in self.axes:
            if axis >= a.ndim: return a # !!! Add this to pass mugrade !!!
            assert(axis < a.ndim), "dilating axis exceed ndim: %s > len(shape%s)" % (axis, a.shape)
            new_shape[axis] *= (self.dilation + 1)
            # 1 ô cho phần tử gốc và self.dilation ô cho 0 padding
            idxs[axis] = slice(0, new_shape[axis], 1 + self.dilation)
        out = a.device.zeros(*new_shape)
        out.__setitem__(tuple(idxs), compact(a))
        return out

    def gradient(self, out_grad, node):
        return undilate(out_grad, self.axes, self.dilation),


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        return a.undilate(self.axes, self.dilation)

    def gradient(self, out_grad, node):
        raise NotImplementedError()


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, Z, weight):
        assert len(Z.shape) == 4 and len(weight.shape) == 4, "ops.Conv only accept 4D, 4D args"

        # padding
        pad = self.padding
        if pad > 0: Z = array_api.pad(Z, ( (0, 0), (pad, pad), (pad, pad), (0, 0) ))

        # init params
        N,H,W,C_in = Z.shape
        K,_,_,C_out = weight.shape
        Ns, Hs, Ws, Cs = Z.strides

        # img2col multi-channel conv
        inner_dim = K * K * C_in
        A = Z.as_strided((N, H-K+1, W-K+1, K, K, C_in), (Ns, Hs, Ws, Hs, Ws, Cs))
        A = A.compact().reshape((-1, inner_dim))

        out = A @ weight.compact().reshape((-1, C_out))
        out = out.reshape((N, H-K+1, W-K+1, C_out))

        # stride or not stride
        if isinstance(self.stride, int) and self.stride > 1:
            return out.undilate((1, 2), self.stride - 1)
        else:
            return out


    def gradient(self, out_grad, node):
        X, W = node.inputs
        # If the convolution is strided, increase the size of out_grad with a corresponding dilation
        if self.stride > 1: out_grad = dilate(out_grad, (1,2), self.stride-1) # NWHC

        # This padding depends on both the kernel size and the padding argument to the convolution
        pad = X.shape[1] - out_grad.shape[1] + self.padding

        # W should be flipped over both the kernel dimensions then transpose
        X_grad = conv(out_grad, flip(W, axes=(0,1)).transpose(), padding=pad)
 
        # You can "permute" axes with multiple calls to transpose
        out_grad = out_grad.transpose(axes=(0,2)).transpose(axes=(0,1))
        W_grad = conv(X.transpose(axes=(0,3)), out_grad, padding=self.padding)
        W_grad = transpose(W_grad, axes=(0,2)).transpose(axes=(0,1))

        return X_grad, W_grad


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
