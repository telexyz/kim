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
        n_inputs = len(node.inputs)
        assert n_inputs == len(out_grad)
        # trả lại gradient thông qua ops.tuple_get_item
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
        in_grads = [init.zeros_like(out_grad) for _ in range(n_inputs - 1)]
        in_grads.insert(self.index, out_grad)
        return make_tensor_tuple(*in_grads),

def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1],

def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, tensors) -> Tensor:
        # https://www.geeksforgeeks.org/python-pytorch-stack-method
        tensor0 = tensors[0]
        shape = list(tensor0.shape)
        shape.insert(self.axis, len(tensors)) # thêm 1 chiều không gian nữa
        idxs = [ slice(0, shape[i], 1) for i in range(len(shape)) ]
        out = NDArray.make(shape, device=tensor0.device)
        for i, tensori in enumerate(tensors):
            assert tensor0.shape == tensori.shape, "stacked tensors must be same shape"
            idxs[self.axis] = slice(i, i+1, 1) # gán dữ liệu vào từng lát cát của chiều self.axis
            out.__setitem__(tuple(idxs), tensori)
        return out

    def gradient(self, out_grad, node):
        return make_tensor_tuple(*split(out_grad, self.axis)),

def stack(args, axis):
    return Stack(axis)(make_tensor_tuple(*args))


import copy
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

    def compute(self, A):
        # print(">>> A:", A.shape, self.axis)
        shape = list(A.shape)
        idxs = [ slice(0,shape[i],1) for i in range(len(shape)) ]
        b_idxs = copy.deepcopy(idxs)

        out = []
        if self.chunks is None:
            del b_idxs[self.axis] # xóa phần tử thứ self.axis của b_idxs
            del  shape[self.axis] # xóa phần tử thứ self.axis của shape
            chunks = A.shape[self.axis]
            offset = 1
        else:
            chunks = self.chunks
            offset = A.shape[self.axis] // chunks
            b_idxs[self.axis] = slice(0, offset, 1)
            shape[self.axis] = offset

        for i in range(chunks):
            idxs[self.axis] = slice(i, i+offset, 1)
            a = A.__getitem__(tuple(idxs))
            b = NDArray.make(shape, device=A.device)
            b.__setitem__(tuple(b_idxs), a)
            out.append(b)
 
        return tuple(out)


    def gradient(self, out_grad, node):
        assert isinstance(out_grad, tuple)
        input_shape = node.inputs[0].shape
        return stack(out_grad, self.axis).reshape(input_shape),


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
        
        axes = self.axes
        if axes is None: axes = ()
        if isinstance(axes, int): axes = (axes,)
        axes = tuple(axes)

        new_shape = out_grad.shape
        for idx in axes:
            new_shape = new_shape[0:idx] + (1,) + new_shape[idx:]        
        # Các thao tác trên chỉ để tính new_shape

        if len(out_grad.shape) == len(a.shape):
            x = out_grad
        else:
            x = reshape(out_grad, new_shape)
        # 
        return broadcast_to(x, a.shape),

def summation(a, axes=None, keepdims=False):
    return Summation(axes, keepdims=keepdims)(a)


class MatMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        ### numpy_backend
        if array_api == np: return a @ b

        # return NDArray(a.numpy() @ b.numpy(), device=a.device) # use numpy for testing
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
        relu_a = Tensor(a > 0, device=out_grad.device)
        return Tensor(out_grad * relu_a, device=out_grad.device),

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
        out.__setitem__(tuple(idxs), a.compact())
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
        # print(">>>", Z.shape, weight.shape)
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
        # print(">>> conv_backward:", out_grad.shape, X.shape, W.shape, self.stride, self.padding)

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
