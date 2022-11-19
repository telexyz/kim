import operator
import math
from functools import reduce
import numpy as np
from . import ndarray_backend_numpy
from . import ndarray_backend_cpu
import os

# math.prod not in Python 3.7
def prod(x):
    return reduce(operator.mul, x, 1)


class BackendDevice:
    """A backend device, wraps the implementation module."""

    def __init__(self, name, mod):
        self.name = name
        self.mod = mod

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return self.name + "()"

    def __getattr__(self, name):
        return getattr(self.mod, name)

    def enabled(self):
        return self.mod is not None

    def zeros(self, *shape, dtype="float32"):
        return NDArray(np.zeros(shape, dtype=dtype), device=self)

    def ones(self, *shape, dtype="float32"):
        return NDArray(np.ones(shape, dtype=dtype), device=self)

    def randn(self, *shape, dtype="float32"):
        '''Return a sample (or samples) from the “standard normal” distribution.'''
        return NDArray(np.random.randn(*shape).astype(dtype), device=self)
        # note: .astype("float32") does work if we're generating a singleton

    def rand(self, *shape, dtype="float32"):
        '''random samples from a uniform distribution over [0, 1).'''
        return NDArray(np.random.rand(*shape).astype(dtype), device=self)

    def one_hot(self, n, i, dtype="float32"):
        return NDArray(np.eye(n, dtype=dtype)[i], device=self)

    def empty(self, shape, dtype="float32"):
        if dtype is None: dtype = "float32"
        assert dtype == "float32"
        return NDArray.make(shape, device=self)

    def full(self, shape, fill_value, dtype="float32"):
        if dtype is None: dtype = "float32"
        assert dtype == "float32"
        arr = self.empty(shape, dtype)
        arr.fill(fill_value)
        return arr


def cuda():
    """Return cuda device"""
    try:
        from . import ndarray_backend_cuda
        return BackendDevice("cuda", ndarray_backend_cuda)

    except ImportError:
        return BackendDevice("cuda", None)


def cuda_triton():
    """Use triton to implement cuda device"""
    import torch # triton use torch to store data in GPU just like cpu_numpy use numpy to store data
    if not torch.cuda.is_available(): return BackendDevice("cuda_triton", None)
    # If torch can use cuda then use triton backend
    from . import ndarray_backend_triton
    return BackendDevice("cuda_triton", ndarray_backend_triton)


def cpu_numpy():
    """Use numpy as a cpu device"""
    return BackendDevice("cpu_numpy", ndarray_backend_numpy)


def cpu():
    """Return cpu device"""
    return BackendDevice("cpu", ndarray_backend_cpu)


PREV_DEVICE=None
def default_device():
    global PREV_DEVICE
    device = os.environ.get("KIM_DEVICE", "cuda_triton")

    if device == "cpu":
        if PREV_DEVICE != device: PREV_DEVICE = device; print("((( USING CPU )))")
        return cpu()

    if device == "cpu_numpy":
        if PREV_DEVICE != device: PREV_DEVICE = device; print("((( USING CPU NUMPY )))")
        return cpu_numpy()

    if device == "cuda":
        if not cuda().enabled(): return cpu()
        if PREV_DEVICE != device: PREV_DEVICE = device; print("((( USING CUDA )))")
        return cuda()

    if device == "cuda_triton":
        if not cuda_triton().enabled(): return cpu()
        if PREV_DEVICE != device: PREV_DEVICE = device; print("((( USING CUDA TRITON )))")
        return cuda_triton()


def all_devices():
    """return a list of all available devices"""
    return [cpu(), cuda(), cpu_numpy(), cuda_triton()]


class NDArray:
    """A generic ND array class that may contain multiple different backends
    i.e., a Numpy backend, a native CPU backend, or a GPU backend.

    This class will only contains those functions that you need to implement
    to actually get the desired functionality for the programming examples
    in the homework, and no more.

    For now, for simplicity the class only supports float32 types, though
    this can be extended if desired.
    """

    def __init__(self, other, device=None):
        """ Create by copying another NDArray, or from numpy """
        if isinstance(other, NDArray):
            # create a copy of existing NDArray
            if device is None: device = other.device
            # tạo một NDArray mới
            array = other.to(device) + 0.0 # this creates a copy
        elif isinstance(other, np.ndarray):
            # create copy from numpy array
            if device is None: device = default_device()
            # tạo một NDArray mới
            array = NDArray.make(other.shape, device=device)
            # copy continuous data from other to array._handle
            array.device.from_numpy(np.ascontiguousarray(other), array._handle)
        else:
            # see if we can create a numpy array from input
            array = NDArray(np.array(other), device=device)
        # khởi tạo self từ NDArray mới được tạo ra
        self._init(array)

    def _init(self, other):
        self._shape   = other._shape
        self._strides = other._strides
        self._offset  = other._offset
        self._device  = other._device
        self._handle  = other._handle

    @staticmethod
    def compact_strides(shape) -> tuple:
        """ Utility function to compute compact strides """
        stride = 1
        res = []
        for i in range(1, len(shape) + 1):
            res.append(stride)
            stride *= shape[-i]
        return tuple(res[::-1])

    @staticmethod
    def make(shape, strides=None, device=None, handle=None, offset=0) -> "NDArray":
        """Create a new NDArray with the given properties. This will allocation the
        memory if handle=None, otherwise it will use the handle of an existing
        array."""

        # tạo NDArray instance mới
        array = NDArray.__new__(NDArray)
        
        # khởi tạo giá trị cho các tham số chưa được khởi tạo
        if strides is None: strides = NDArray.compact_strides(shape)
        if device is None: device = default_device()
        if handle is None: handle = device.Array(prod(shape)) # tạo vùng nhớ mới trong device
        
        # gán các tham số vào thuộc tính của instance mới
        array._shape = tuple(shape)
        array._strides = tuple(strides)
        array._offset = offset
        array._device = device
        array._handle = handle

        return array
    
    ### String representations
    def __repr__(self):
        return "NDArray(" + self.numpy().__str__() + f", device={self.device})"

    def __str__(self): return self.numpy().__str__()

    ### Properies
    @property
    def shape(self): return self._shape

    @property
    def strides(self): return self._strides

    @property
    def device(self): return self._device

    @property
    def dtype(self): return "float32" # only support float32 for now

    @property
    def ndim(self): return len(self._shape) # Return number of dimensions.
        
    @property
    def size(self): return prod(self._shape)

    @property
    def flat(self) -> "NDArray": return self.reshape((self.size,))

    ### Basic array manipulation
    def fill(self, value):
        self._device.fill(self._handle, value)

    def to(self, device) -> "NDArray":
        """ Convert between devices,
        using to/from numpy calls as the unifying bridge. """
        if device == self.device:
            return self
        else:
            return NDArray(self.numpy(), device=device)

    def numpy(self) -> np.ndarray:
        """ convert to a numpy array """
        x = self.device.to_numpy(self._handle, self._shape, self._strides, self._offset)
        return x

    def is_compact(self) -> bool:
        """Return true if array is compact in memory and internal size equals product
        of the shape dimensions"""
        return (self._strides == self.compact_strides(self._shape)
            and prod(self.shape) == self._handle.size)

    def compact(self) -> "NDArray":
        """ Convert a matrix to be compact """
        if self.is_compact():
            return self
        else:
            '''Tạo một compact NDArray mới và copy dữ liệu sang'''
            out = NDArray.make(self.shape, device=self.device)
            # print(">>> Compact", self.shape, self.strides, self._offset);
            # raise ValueError
            self.device.compact(self._handle, out._handle, 
                self._shape, self._strides, self._offset)
            return out


    def as_strided(self, shape: tuple, strides: tuple) -> "NDArray":
        """ Restride the matrix without copying memory. """
        assert len(shape) == len(strides)
        return NDArray.make(shape, strides=strides,
            device=self.device, handle=self._handle)


    def reshape(self, new_shape: tuple) -> "NDArray":
        """
        Reshape the matrix without copying memory. This will return a matrix
        that corresponds to a reshaped array but points to the same memory as
        the original array.
        Raises:
            ValueError if product of current shape is not equal to the product
            of the new shape, or if the matrix is not compact.
        Args:
            new_shape (tuple): new shape of the array
        Returns:
            NDArray: reshaped array; this will point to the same memory as the original NDArray.
        """
        # print(">>>", new_shape)
        if len(new_shape) == 2 and new_shape[0] == -1 and new_shape[1] > 0:
            new_shape = (prod(self.shape) // new_shape[1], new_shape[1])
        # if prod(self.shape) != prod(new_shape): raise ValueError
        assert prod(self.shape) == prod(new_shape), "%s shape != new_shape" % (new_shape)
        # assert self.is_compact(), "%s is not compact" % ("self") # (self)
        new_strides = self.compact().compact_strides(new_shape)
        return self.as_strided(new_shape, new_strides)


    def permute(self, new_axes: tuple) -> "NDArray":
        """
        Permute order of the dimensions. new_axes describes a permutation of the
        existing axes, so e.g.:

          - If we have an array with dimension "BHWC" then .permute((0,3,1,2))
            would convert this to "BCHW" order.

          - For a 2D array, .permute((1,0)) would transpose the array.

        Like reshape, this operation should not copy memory, but achieves the
        permuting by just adjusting the shape/strides of the array.  That is,
        it returns a new array that has the dimensions permuted as desired, but
        which points to the same memory as the original array.

        Args:
            new_axes (tuple): permutation order of the dimensions

        Returns:
            NDarray : new NDArray object with permuted dimensions, pointing
            to the same memory as the original NDArray (i.e., just shape and
            strides changed).
        """
        new_shape = [self.shape[i] for i in new_axes]
        new_strides = [self.strides[i] for i in new_axes]
        return self.as_strided(new_shape, new_strides)

    def swapaxes(self, a1, a2) -> "NDArray":
        new_shape = list(self.shape)
        new_strides = list(self.strides)

        new_shape[a1] = self.shape[a2]
        new_shape[a2] = self.shape[a1]

        new_strides[a1] = self.strides[a2]
        new_strides[a2] = self.strides[a1]

        return self.as_strided(new_shape, new_strides)


    def broadcast_to(self, new_shape) -> "NDArray":
        """
        Broadcast an array to a new shape. new_shape's elements must be the
        same as the original shape, except for dimensions in the self where
        the size = 1 (which can then be broadcast to any size).  As with the
        previous calls, this will not copy memory, and just achieves
        broadcasting by manipulating the strides.

        Raises:
            assertion error if new_shape[i] != shape[i] for all i where
            shape[i] != 1
        Args:
            new_shape (tuple): shape to broadcast to
        Returns:
            NDArray: the new NDArray object with the new broadcast shape; should
            point to the same memory as the original array.
        """
        if len(self.shape) != len(new_shape): raise ValueError("Only broadcast to same shape len")
        new_strides = []
        for i in range(len(self.shape)):
            if self.shape[i] != 1 and self.shape[i] != new_shape[i]: raise ValueError
            if self.shape[i] == 1:
                new_strides.append(0)
            else:
                new_strides.append(self.strides[i])
        return self.as_strided(new_shape, tuple(new_strides))

    ### Get and set elements

    def process_slice(self, sl, dim) -> slice:
        """ Convert a slice to an explicit start/stop/step """
        start, stop, step = sl.start, sl.stop, sl.step

        if start == None: start = 0
        if stop == None: stop = self.shape[dim]
        if step == None: step = 1

        if start < 0: start += self.shape[dim]
        if stop < 0: stop += self.shape[dim]

        # we're not gonna handle negative strides and that kind of thing
        assert stop > start, "Start must be less than stop"
        assert step > 0, "No support for negative increments"
        return slice(start, stop, step)

    def __getitem__(self, idxs) -> "NDArray":
        """
        The __getitem__ operator in Python allows us to access elements of our
        array.  When passed notation such as a[1:5,:-1:2,4,:] etc, Python will
        convert this to a tuple of slices and integers (for singletons like the
        '4' in this example).  Slices can be a bit odd to work with (they have
        three elements .start .stop .step), which can be None or have negative
        entries, so for simplicity we wrote the code for you to convert these
        to always be a tuple of slices, one of each dimension.
        For this tuple of slices, return an array that subsets the desired
        elements.  

        ** As before, this can be done entirely through compute a new
        shape, stride, and offset for the new "view" into the original array,
        pointing to the same memory **

        Raises:
            AssertionError if a slice has negative size or step, or if number
            of slices is not equal to the number of dimension (the stub code
            already raises all these errors.
        Args:
            idxs tuple: (after stub code processes), a tuple of slice elements
            corresponding to the subset of the matrix to get
        Returns:
            NDArray: a new NDArray object corresponding to the selected
            subset of elements.  As before, this should not copy memory but just
            manipulate the shape/strides/offset of the new array, referencing
            the same array as the original one.
        """

        # handle singleton as tuple, everything as slices
        if not isinstance(idxs, tuple): idxs = (idxs,)
        idxs = tuple(
            [
                self.process_slice(s, i) if isinstance(s, slice) else slice(s, s+1, 1)
                for i, s in enumerate(idxs)
            ]
        )
        assert len(idxs) == self.ndim, "Need indexes equal to number of dimensions"

        new_offset = 0
        new_shape = []
        new_strides = []

        for i in range(len(idxs)):
            idx = idxs[i]

            '''Raises:
                AssertionError if a slice has negative size or step, or if number
                of slices is not equal to the number of dimension (the stub code
                already raises all these errors.'''
            if idx.start > idx.stop or idx.step <= 0: raise AssertionError
            
            new_shape.append(math.ceil((idx.stop - idx.start) / idx.step))
            new_strides.append(self.strides[i] * idx.step)
            new_offset += idx.start * self.strides[i]
            # print(">>>", i, idx, self.strides[i], new_offset, new_shape, new_strides)
        # print("__getitem__", idxs)

        '''Returns:
            NDArray: a new NDArray object corresponding to the selected
            subset of elements.  As before, this should not copy memory but just
            manipulate the shape/strides/offset of the new array, referencing
            the same array as the original one.'''
        assert len(new_shape) == len(new_strides)
        return NDArray.make(
            tuple(new_shape), strides=tuple(new_strides), device=self.device, 
            handle=self._handle, offset=new_offset
        )


    def __setitem__(self, idxs, other):
        """Set the values of a view into an array, using the same semantics
        as __getitem__()."""
        view = self.__getitem__(idxs)
        # print(">>> view:", view.shape)
        if isinstance(other, NDArray):
            assert prod(view.shape) == prod(other.shape)
            self.device.ewise_setitem(
                other.compact()._handle,
                view._handle,
                view._shape,
                view._strides,
                view._offset,
            )
        else:
            self.device.scalar_setitem(
                prod(view.shape),
                other,
                view._handle,
                view._shape,
                view._strides,
                view._offset,
            )

    ### Collection of elementwise and scalar function: add, multiply, boolean, etc

    def ewise_or_scalar(self, other, ewise_func, scalar_func) -> "NDArray":
        """Run either an element-wise or scalar version of a function,
        depending on whether "other" is an NDArray or scalar
        """
        out = NDArray.make(self.shape, device=self.device)
        if isinstance(other, NDArray):
            assert self.shape == other.shape, "operation needs two equal-sized arrays"
            ewise_func(self.compact()._handle, other.compact()._handle, out._handle)
        else:
            scalar_func(self.compact()._handle, other, out._handle)
        return out

    def __add__(self, other) -> "NDArray":
        return self.ewise_or_scalar(
            other, self.device.ewise_add, self.device.scalar_add)

    def __mul__(self, other) -> "NDArray":
        return self.ewise_or_scalar(
            other, self.device.ewise_mul, self.device.scalar_mul)

    __radd__ = __add__
    __rmul__ = __mul__

    def __sub__(self, other) -> "NDArray": return self + (-other)
    def __rsub__(self, other) -> "NDArray": return other + (-self)
    def __neg__(self) -> "NDArray": return self * (-1)


    def __truediv__(self, other) -> "NDArray":
        return self.ewise_or_scalar(
            other, self.device.ewise_div, self.device.scalar_div)

    def __pow__(self, other) -> "NDArray":
        out = NDArray.make(self.shape, device=self.device)
        self.device.scalar_power(self.compact()._handle, other, out._handle)
        return out

    def maximum(self, other) -> "NDArray":
        return self.ewise_or_scalar(
            other, self.device.ewise_maximum, self.device.scalar_maximum)

    ### Binary operators all return (0.0, 1.0) floating point values, could of course be optimized
    def __eq__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_eq, self.device.scalar_eq)

    def __ge__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_ge, self.device.scalar_ge)

    def __ne__(self, other):
        return 1 - (self == other)

    def __gt__(self, other):
        return (self >= other) * (self != other)

    def __lt__(self, other):
        return 1 - (self >= other)

    def __le__(self, other):
        return 1 - (self > other)

    ### Elementwise functions

    def log(self) -> "NDArray":
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_log(self.compact()._handle, out._handle)
        return out

    def exp(self) -> "NDArray":
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_exp(self.compact()._handle, out._handle)
        return out

    def tanh(self) -> "NDArray":
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_tanh(self.compact()._handle, out._handle)
        return out


    ### Matrix multiplication
    def __matmul__(self, other) -> "NDArray":
        """Matrix multiplication of two arrays.  This requires that both arrays
        be 2D (i.e., we don't handle batch matrix multiplication), and that the
        sizes match up properly for matrix multiplication.

        In the case of the CPU backend, you will implement an efficient "tiled"
        version of matrix multiplication for the case when all dimensions of
        the array are divisible by self.device.__tile_size__.  In this case,
        the code below will re-stride and compact the matrix into tiled form,
        and then pass to the relevant CPU backend.  For the CPU version we will
        just fall back to the naive CPU implementation if the array shape is not
        a multiple of the tile size

        The GPU (and numpy) versions don't have any tiled version (or rather,
        the GPU version will just work natively by tiling any input size).
        """

        assert self.ndim == 2 and other.ndim == 2, "matmul 2D arrays only"
        assert self.shape[1] == other.shape[0]

        m, n, p = self.shape[0], self.shape[1], other.shape[1]

        # if the matrix is aligned, use tiled matrix multiplication
        if hasattr(self.device, "matmul_tiled") and all(
            d % self.device.__tile_size__ == 0 for d in (m, n, p)
        ):

            def tile(a, tile):
                return a.as_strided(
                    (a.shape[0] // tile, a.shape[1] // tile, tile, tile),
                    (a.shape[1] * tile, tile, a.shape[1], 1),)

            t = self.device.__tile_size__
            a = tile(self.compact(), t).compact()
            b = tile(other.compact(), t).compact()
            out = NDArray.make((a.shape[0], b.shape[1], t, t), device=self.device)
            self.device.matmul_tiled(a._handle, b._handle, out._handle, m, n, p)

            return (
                out.permute((0, 2, 1, 3))
                .compact()
                .reshape((self.shape[0], other.shape[1]))
            )

        else:
            out = NDArray.make((m, p), device=self.device)
            self.device.matmul(
                self.compact()._handle, other.compact()._handle, out._handle, m, n, p
            )
            return out


    ### Reductions, i.e., sum/max over all element or over given axis
    def reduce_view_out(self, axis, keepdims=False):
        """ Return a view to the array set up for reduction functions and output array. """
        if axis is None:
            view = self.compact().reshape((1,) * (self.ndim - 1) + (prod(self.shape),))
            # out = NDArray.make((1,), device=self.device)
            if keepdims:
                out = NDArray.make((1,) * self.ndim, device=self.device)
            else:
                out = NDArray.make((), device=self.device)
            return view, out

        ### axis is not None
        if isinstance(axis, (tuple, list)):
            axes = tuple(axis)
        else:
            axes = (axis,)

        # Create view
        fixed_axes = tuple([a for a in range(self.ndim) if a not in axes])
        new_shape = tuple([self.shape[x] for x in fixed_axes])
        new_shape = new_shape + (prod([self.shape[x] for x in axes]),)

        view = self.permute(fixed_axes + axes)
        if len(axes) > 1: view = view.compact().reshape(new_shape)

        # Create out
        if keepdims:
            new_shape = [1 if i in axes else s for i, s in enumerate(self.shape)]
        else:
            new_shape = [s for i, s in enumerate(self.shape) if i not in axes]

        out = NDArray.make(tuple(new_shape), device=self.device)

        return view, out


    def sum(self, axis=None, keepdims=False):
        # print(">>> sum:", axis, keepdims)
        view, out = self.reduce_view_out(axis, keepdims=keepdims)
        self.device.reduce_sum(view.compact()._handle, out._handle, view.shape[-1])
        return out


    def max(self, axis=None, keepdims=False):
        view, out = self.reduce_view_out(axis, keepdims=keepdims)
        self.device.reduce_max(view.compact()._handle, out._handle, view.shape[-1])
        return out


    def flip(self, axes):
        """
        Flip this ndarray along the specified axes.
        Note: compact() before returning.
        """
        ### BEGIN YOUR SOLUTION
        x = self.compact()
        new_offset = 0
        new_strides = list(x.strides)
        for a in axes:
            new_strides[a] *= -1
            # (shape[0]) - 1)*shape[1]*shape[2] => a = 0
            offset = x.shape[a] - 1
            for b in range(len(x.shape) - a - 1): offset *= x.shape[a + b + 1]
            # print(">>> offset:", offset)
            new_offset += offset
        out = NDArray.make(
            x.shape, strides=tuple(new_strides), device=x.device, 
            handle=x._handle, offset=new_offset
        )
        return out.compact()
        ### END YOUR SOLUTION

    def undilate(a, axes, dilation):
        new_shape = list(a.shape)
        idxs = [slice(0, a.shape[i], 1) for i in range(len(a.shape))]
        for axis in axes:
            new_shape[axis] //= (dilation + 1)
            idxs[axis] = slice(0, a.shape[axis], dilation + 1)
        return a.compact().__getitem__(tuple(idxs))


    def pad(self, axes):
        """
        Pad this ndarray by zeros by the specified amount in `axes`,
        which lists for _all_ axes the left and right padding amount, e.g.,
        axes = ( (0, 0), (1, 1), (0, 0)) pads the middle axis with a 0 on the left and right side.
        """
        ### BEGIN YOUR SOLUTION
        idx = ()
        shape = list(self.shape)
        for i in range(len(axes)):
            l = axes[i][0]
            shape[i] += l
            idx = idx + (slice(l, shape[i], 1), )
            shape[i] += axes[i][1]

        out = self.device.zeros(*shape)
        out.__setitem__(idx, self.compact())
        return out
        ### END YOUR SOLUTION


def array(a, dtype="float32", device=None):
    """ Convenience methods to match numpy a bit more closely."""
    if dtype is None: dtype = "float32"
    assert dtype == "float32"
    return NDArray(a, device=device)


def empty(shape, dtype="float32", device=None):
    if device is None: device = default_device()
    return device.empty(shape, dtype)


def full(shape, fill_value, dtype="float32", device=None):
    if device is None: device = default_device()
    return device.full(shape, fill_value, dtype)


def broadcast_to(array, new_shape):
    return array.broadcast_to(new_shape)

def reshape(array, new_shape):
    return array.reshape(new_shape)


def maximum(a, b):
    return a.maximum(b)


def max(a, axis=None):
    return a.max(axis)


def log(a):
    return a.log()


def exp(a):
    return a.exp()


def tanh(a):
    return a.tanh()


def flip(a, axes):
    return a.flip(axes)


def summation(a, axis=None, keepdims=False):
    return a.sum(axis=axis, keepdims=keepdims)


def sum(a, axis=None, keepdims=False):
    return a.sum(axis=axis, keepdims=keepdims)
