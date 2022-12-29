import kim
from typing import List, Optional, NamedTuple, Tuple, Union
import numpy
from .tensor_tuple import TensorTuple, TensorTupleOp
from .backend_selection import Device, array_api, NDArray, default_device

# namespace để chứa config và counters liên quan tới computational graph
class CompGraph:
    LAZY_MODE = False
    NODE_COUNT = 0
    SAVE_MEM = True

####################################
####### Tensor và TensorOp   #######
####### Cốt tủy của AutoGrad #######
####################################

class TensorOp:
    def __call__(self, *args):
        return Tensor.make_from_op(self, args)


class Tensor:
    def __repr__(self):
        return "kim.Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        return self.realize_cached_data().__str__()
        # return self.__repr__()

    def __del__(self):
        CompGraph.NODE_COUNT -= 1
    
    cached_data: Optional[NDArray]
    grad: "Tensor" # lưu out_grad gradient của node
    requires_grad: bool
    visited=None

    op: TensorOp
    inputs: List

    def realize_cached_data(self) -> NDArray:
        if self.cached_data is None:
            self.cached_data = self.op.compute(
                *[x.realize_cached_data() for x in self.inputs]
            )
        return self.cached_data
    
    def numpy(self):
        data = self.realize_cached_data()
        if array_api is numpy: return data
        if isinstance(data, tuple): return [x.numpy() for x in data]
        return data.numpy()

    def assign_params_and_record_creation(
        self, op: Optional[TensorOp],
        inputs: List["Tensor"],
        cached_data: Optional[NDArray] = None,
        requires_grad: bool = False
    ):
        CompGraph.NODE_COUNT += 1
        self.op = op
        self.inputs = inputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad or any(x.requires_grad for x in inputs)

    def __init__(self, array: Union["Tensor", NDArray], 
        device: Optional[Device] = None,
        dtype=None,
        requires_grad=True
    ):
        if isinstance(array, Tensor):
            if device is None: device = array.device
            if dtype is None: dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                # fall back, copy through numpy conversion
                cached_data = make_array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        else:
            if device is None: device = default_device()
            cached_data = make_array_from_numpy(array, device=device, dtype=dtype)
        
        self.assign_params_and_record_creation(
            op=None, inputs=[],
            cached_data=cached_data,
            requires_grad=requires_grad
        )
    
    @staticmethod
    def make_from_op(op: TensorOp, inputs: List["Tensor"]) -> "Tensor":
        tensor = Tensor.__new__(Tensor) # dùng __new__(cls, ..) để bỏ qua __init__
        tensor.assign_params_and_record_creation(op=op, inputs=inputs)
        if not CompGraph.LAZY_MODE:
            if not tensor.requires_grad:
                return tensor.detach() # tách tensor khỏi đồ thị tính toán
            tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(
        array: Union["Tensor", NDArray], 
        requires_grad: bool = False
    ) -> "Tensor":
        tensor = Tensor.__new__(Tensor)
        if isinstance(array, Tensor): array = array.realize_cached_data()
        tensor.assign_params_and_record_creation(
            op=None, inputs=[],
            cached_data=array,
            requires_grad=requires_grad
        )
        return tensor

    
    def detach(self):
        return Tensor.make_const(self.realize_cached_data())

    def backward(self, out_grad: Optional["Tensor"] = None):
        if out_grad is None: out_grad = kim.init.ones(*self.shape, dtype=self.dtype, device=self.device)
        compute_gradient_from(self, out_grad)

    @property
    def data(self): return self.detach()

    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        self.cached_data = value.realize_cached_data()

    @property
    def shape(self): return self.realize_cached_data().shape

    @property
    def dtype(self): return self.realize_cached_data().dtype

    @property
    def device(self):
        if array_api is numpy: return default_device()
        else: return self.realize_cached_data().device

    """ Syntax sugar, không có không sao
    """
    def __add__(self, other):
        if isinstance(other, Tensor): return kim.ops.EWiseAdd()(self, other)
        else: return kim.ops.AddScalar(other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor): return kim.ops.EWiseMul()(self, other)
        else: return kim.ops.MulScalar(other)(self)

    def __pow__(self, other):
        if isinstance(other, Tensor): raise NotImplementedError()
        else: return kim.ops.PowerScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return kim.ops.EWiseAdd()(self, kim.ops.Negate()(other))
        else:
            return kim.ops.AddScalar(-other)(self)

    def __rsub__(self, other):
        if isinstance(other, Tensor):
            return kim.ops.EWiseAdd()(kim.ops.Negate()(self), other)
        else:
            return kim.ops.AddScalar(other)(-self)

    def __truediv__(self, other):
        if isinstance(other, Tensor): return kim.ops.EWiseDiv()(self, other)
        else: return kim.ops.DivScalar(other)(self)

    def __neg__(self):
        return kim.ops.Negate()(self)

    def __matmul__(self, other):
        return kim.ops.MatMul()(self, other)

    def matmul(self, other):
        return kim.ops.MatMul()(self, other)

    def sum(self, axes=None):
        return kim.ops.Summation(axes)(self)

    def reshape(self, shape):
        return kim.ops.Reshape(shape)(self)

    def broadcast_to(self, shape):
        return kim.ops.BroadcastTo(shape)(self)

    def transpose(self, axes=None):
        return kim.ops.Transpose(axes)(self)

    def permute(self, axes=None):
        return kim.ops.Permute(axes)(self)

    __radd__ = __add__
    __rmul__ = __mul__

##############################
####### Helper Methods #######
##############################

def make_array_from_numpy(numpy_array, device, dtype):
    if array_api is numpy: return numpy.array(numpy_array, dtype=dtype)
    else: return array_api.array(numpy_array, device=device, dtype="float32")


def compute_gradient_from(out_tensor: Tensor, out_grad: Tensor):
    output_grads: Dict[Tensor, List[Tensor]] = {}
    output_grads[out_tensor] = [out_grad]
    reverse_topo_order = reversed(find_topo_sort([out_tensor]))

    for node in reverse_topo_order:
        if not node.requires_grad: continue

        node.grad = output_grads[node].pop()
        for grad in output_grads[node]: node.grad += grad # accummulate other grads

        if CompGraph.SAVE_MEM:
            node.grad = node.grad.detach() # will save a lot of (GPU) memory

        if node.op is not None:
            grads = node.op.gradient(node.grad, node)
            for k, inp in enumerate(node.inputs):
                grad = grads[k]
                try: output_grads[inp].append(grad)
                except KeyError: output_grads[inp] = [grad]


import random
def find_topo_sort(nodes):
    visited = random.randint(0,99999)
    topo_order = []

    def topo_sort_dfs(node):
        node.visited = visited
        for inp in node.inputs:
            if inp.visited != visited:
                topo_sort_dfs(inp)
        topo_order.append(node)

    for node in nodes: topo_sort_dfs(node)
    return topo_order
