import numpy as np
import ctypes

# Utils to switch between ndarray_backend and numpy_backend
def as_numpy(x):
    if isinstance(x, np.ndarray): return x
    if isinstance(x, np.float32): return x
    if isinstance(x, np.int64): return x
    return x.numpy()

import operator
from functools import reduce
# math.prod not in Python 3.7
def prod(x):
    return reduce(operator.mul, x, 1)

'''
https://github.com/dlsyscourse/hw4/blob/main/hw4.ipynb#Flipping-ndarrays-&-FlipOp
'''
# reads off the underlying data array in order (i.e., offset 0, offset 1, ..., offset n) i.e., ignoring strides
def raw_data(X):
    X = np.array(X) # copy, thus compact X
    return np.frombuffer(ctypes.string_at(X.ctypes.data, X.nbytes), dtype=X.dtype, count=X.size)

# Xold and Xnew should reference the same underlying data
def offset(Xold, Xnew):
    assert Xold.itemsize == Xnew.itemsize
    # compare addresses to the beginning of the arrays
    return (Xnew.ctypes.data - Xold.ctypes.data)//Xnew.itemsize

def strides(X):
    return ', '.join([str(x//X.itemsize) for x in X.strides])

def format_array(X, shape):
    assert len(shape) == 3, "I only made this formatting work for ndims = 3"
    def chunks(l, n):
        n = max(1, n)
        return (l[i:i+n] for i in range(0, len(l), n))
    a = [str(x) if x >= 10 else ' ' + str(x) for x in X]
    a = ['(' + ' '.join(y) + ')' for y in [x for x in chunks(a, shape[-1])]]
    a = ['|' + ' '.join(y) + '|' for y in [x for x in chunks(a, shape[-2])]]
    return '  '.join(a)

def inspect_array(X, *, is_a_copy_of):
    # compacts X, then reads it off in order
    print('Data: %s' % format_array(raw_data(X), X.shape))
    # compares address of X to copy_of, thus finding X's offset
    print('Offset: %s' % offset(is_a_copy_of, X))
    print('Strides: %s' % strides(X))

'''
>>> A = np.arange(1, 25).reshape(3, 2, 4) # 24 elems
>>> kim.utils.inspect_array(A, is_a_copy_of=A)
Data: |( 1  2  3  4) ( 5  6  7  8)|
      |( 9 10 11 12) (13 14 15 16)|
      |(17 18 19 20) (21 22 23 24)|
Offset: 0
Strides: 8, 4, 1

>>> kim.utils.inspect_array(np.flip(A, (2,)), is_a_copy_of=A)
Data: |( 4  3  2  1) ( 8  7  6  5)|
      |(12 11 10  9) (16 15 14 13)|
      |(20 19 18 17) (24 23 22 21)|
Offset: 3 => (shape[2]) - 1)*1
Strides: 8, 4, -1

Với [0,0,0] thì strides ko có tác dụng, luôn trỏ tới vị trí đầu tiên =>
offset trỏ tới vị trí đầu tiên

>>> kim.utils.inspect_array(np.flip(A, (1,)), is_a_copy_of=A)
Data: |( 5  6  7  8) ( 1  2  3  4)|
      |(13 14 15 16) ( 9 10 11 12)|
      |(21 22 23 24) (17 18 19 20)|
Offset: 4 => (shape[1] - 1)*shape[2]
Strides: 8, -4, 1

>>> kim.utils.inspect_array(np.flip(A, (0,)), is_a_copy_of=A)
Data: |(17 18 19 20) (21 22 23 24)|
      |( 9 10 11 12) (13 14 15 16)|
      |( 1  2  3  4) ( 5  6  7  8)|
Offset: 16 => (shape[0]) - 1)*shape[1]*shape[2]
Strides: -8, 4, 1



>>> kim.utils.inspect_array(np.flip(A, (0,1,2)), is_a_copy_of=A)
Data: |(24 23 22 21) (20 19 18 17)|
      |(16 15 14 13) (12 11 10  9)|
      |( 8  7  6  5) ( 4  3  2  1)|
Offset: 23
Strides: -8, -4, -1

>>> kim.utils.inspect_array(np.flip(A, (1,2)), is_a_copy_of=A)
Data: |( 8  7  6  5) ( 4  3  2  1)|
      |(16 15 14 13) (12 11 10  9)|
      |(24 23 22 21) (20 19 18 17)|
Offset: 7
Strides: 8, -4, -1

>>> kim.utils.inspect_array(np.flip(A, (0,1)), is_a_copy_of=A)
Data: |(21 22 23 24) (17 18 19 20)|
      |(13 14 15 16) ( 9 10 11 12)|
      |( 5  6  7  8) ( 1  2  3  4)|
Offset: 20
Strides: -8, -4, 1
'''