import numpy as np # ndarray use numpy as intermediate medium
import triton
import triton.language as tl
import torch # triton use torch as a CUDA data storage

cuda = torch.device("cuda") # to init torch.Tensor in CUDA

import operator
from functools import reduce
def prod(x): return reduce(operator.mul, x, 1) # utility function

class Array:
    ''' Like triton, our Array class use torch.Tensor as a data storage too '''
    def __init__(self, size: int) -> torch.Tensor:
        self.array = torch.empty(size, dtype=torch.float32, device=cuda)

    @property
    def size(self) -> int: return self.array.size()[0]


def to_numpy(a, shape, strides, offset) -> np.ndarray:
    return torch.as_strided(a.array, shape, strides, storage_offset=offset).cpu().numpy()

def from_numpy(a, out): out.array[:] = torch.from_numpy(a.flatten())

def fill(out, val): out.array.fill_(val)

def compact(a, out, shape, strides, offset):
    from_numpy(to_numpy(a, shape, strides, offset).flatten(), out)

''' Triton ops
'''
def scalar_setitem(size, val, out, shape, strides, offset):
    # super simple triton kernel integration, just to show how to use triton with our backend
    a = Array(prod(shape)) # init an empty array, then set it's all elems to val
    grid = lambda meta: (triton.cdiv(a.size, meta['BLOCK_SIZE']),)
    simple_scalar_setitem_kernel[grid](a.array, a.size, val, BLOCK_SIZE=512)
    # then using ewise_setitem to assign them to out
    ewise_setitem(a, out, shape, strides, offset)


''' Use Torch functions to pass the tests first
'''
def ewise_setitem(a, out, shape, strides, offset):
    torch.as_strided(out.array, shape, strides, offset)[:] = a.array.reshape(shape)

def ewise_add(a, b, out): out.array[:] = a.array + b.array

def scalar_add(a, val, out): out.array[:] = a.array + val

def ewise_mul(a, b, out): out.array[:] = a.array * b.array

def scalar_mul(a, val, out): out.array[:] = a.array * val

def ewise_div(a, b, out): out.array[:] = a.array / b.array

def scalar_div(a, val, out): out.array[:] = a.array / val

def scalar_power(a, val, out): out.array[:] = a.array ** val

def ewise_maximum(a, b, out): out.array[:] = torch.maximum(a.array, b.array)

def scalar_maximum(a, val, out): out.array[:] = torch.clamp(a.array, min=val)

def ewise_eq(a, b, out): out.array[:] = (a.array == b.array)

def scalar_eq(a, val, out): out.array[:] = (a.array == val)

def ewise_ge(a, b, out): out.array[:] = (a.array >= b.array)

def scalar_ge(a, val, out): out.array[:] = (a.array >= val)

def ewise_log(a, out): out.array[:] = torch.log(a.array)

def ewise_exp(a, out): out.array[:] = torch.exp(a.array)

def ewise_tanh(a, out): out.array[:] = torch.tanh(a.array)

def matmul(a, b, out, m, n, p):
    out.array[:] = (a.array.reshape(m, n) @ b.array.reshape(n, p)).reshape(-1)

def reduce_max(a, out, reduce_size):
    out.array[:] = a.array[:].reshape(-1, reduce_size).max(axis=1).values

def reduce_sum(a, out, reduce_size):
    out.array[:] = a.array[:].reshape(-1, reduce_size).sum(axis=1)


######################
#                    #
#   TRITON KERNELS   #
#                    #
######################

'''This kernel simply set all value of an output vector to a scalar value'''
@triton.jit
def simple_scalar_setitem_kernel(
    output_ptr,  # *Pointer* to output vector.
    n_elements,  # Size of the vector.
    value,
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    output = tl.load(output_ptr + offsets, mask=mask)
    output = value
    tl.store(output_ptr + offsets, output, mask=mask)
