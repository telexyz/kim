import kim
from typing import List, Optional, NamedTuple, Tuple, Union
import numpy

from .backend_selection import Device, array_api, NDArray, default_device

class State:
    LAZY_MODE = False
    TENSOR_COUNT = 0
    MAX_BACKWARD_TENSOR_COUNT = 0

####################################
####### Tensor và TensorOp   #######
####### Cốt tủy của AutoGrad #######
####################################

class TensorOp:
    def __call__(self, *args):
        return Tensor.make_from_op(self, args)
    def compute():
        raise NotImplementedError()
    def gradient(self, out_grad: "Tensor", node: "Tensor") -> Tuple["Tensor"]:
        raise NotImplementedError()

class TensorTupleOp(TensorOp):
    def __call__(self, *args):
        return TensorTuple.make_from_op(self, args)


class Tensor:
    def __repr__(self):
        return "kim.Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        return self.realize_cached_data().__str__()

    def __del__(self):
        State.TENSOR_COUNT -= 1
    
    cached_data: Optional[NDArray]
    grad: "Tensor" # lưu out_grad gradient của node
    requires_grad: bool

    op: TensorOp
    inputs: List["Tensor"]

    def realize_cached_data(self) -> NDArray:
        if self.cached_data is None:
            self.cached_data = self.op.compute(
                *[x.realize_cached_data() for x in self.inputs]
            )
        return self.cached_data
    
    def numpy(self):
        data = self.realize_cached_data()
        if array_api is numpy: return data
        else: return data.numpy()

    def assign_params_and_record_creation(
        self, op: Optional[TensorOp],
        inputs: List["Tensor"],
        cached_data: Optional[NDArray] = None,
        requires_grad: bool = False
    ):
        State.TENSOR_COUNT += 1
        if not requires_grad:
            requires_grad = any(x.requires_grad for x in inputs)
        self.op = op
        self.inputs = inputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad

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
            if device is None: device = default_device()
            cached_data = get_array_from_numpy(array, device=device, dtype=dtype)
        
        self.assign_params_and_record_creation(
            op=None, inputs=[],
            cached_data=cached_data,
            requires_grad=requires_grad
        )
    
    @staticmethod
    def make_from_op(op: TensorOp, inputs: List["Tensor"]) -> "Tensor":
        tensor = Tensor.__new__(Tensor) # dùng __new__(cls, ..) để bỏ qua __init__
        tensor.assign_params_and_record_creation(op=op, inputs=inputs)
        if not State.LAZY_MODE:
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
        if out_grad is None:
            out_grad = Tensor(default_device().ones(*self.shape, dtype="float32"))
        compute_gradient_of(self, out_grad)

    @property
    def data(self): return self.detach()

    @data.setter
    def data(self, value):
        # print(">>>", value)
        assert isinstance(value, Tensor)
        # assert value.dtype == self.dtype, "%s %s" % (value.dtype, self.dtype)
        self.cached_data = value.realize_cached_data().astype(self.dtype)

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

    __radd__ = __add__
    __rmul__ = __mul__


################################
####### Cấu trúc dữ liệu #######
####### trên Tensor      #######
################################

class TensorTuple(Tensor):
    def __len__(self):
        return len(self.realize_cached_data())

    def __getitem__(self, index: int):
        return kim.ops.tuple_get_item(self, index)

    def tuple(self):
        return tuple([x for x in self])

    def __repr__(self):
        return "kim.TensorTuple" + str(self.tuple())

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        assert isinstance(other, TensorTuple)
        assert len(self) == len(other)
        return kim.ops.make_tuple(*[self[i] + other[i] for i in range(len(self))])

    def detach(self):
        return Tuple.make_const(self.realize_cached_data())


##############################
####### Helper Methods #######
##############################

def get_array_from_numpy(numpy_array, device, dtype):
    if array_api is numpy: return numpy.array(numpy_array, dtype=dtype)
    else: return array_api.array(numpy_array, device=device, dtype=dtype)


def compute_gradient_of(output_tensor: Tensor, out_grad: Tensor):
    output_grads: Dict[Tensor, List[Tensor]] = {}
    output_grads[output_tensor] = [out_grad]
    reverse_topo_order = reversed(find_topo_sort([output_tensor]))

    for node in reverse_topo_order:
        if not node.requires_grad: continue
        node.grad = sum(x for x in output_grads[node])

        # Detach grad from computational graph to save memory
        if State.TENSOR_COUNT > State.MAX_BACKWARD_TENSOR_COUNT:
            node.grad = node.grad.detach()

        # print(">>>", node.op) # bắt lỗi grad không phải float32
        # assert node.grad.dtype == "float32", "%s %s" % (node.grad.dtype, node.dtype)

        if node.op:
            grads = node.op.gradient(node.grad, node)
            for k in range(len(node.inputs)):
                try: output_grads[node.inputs[k]].append(grads[k])
                except KeyError: output_grads[node.inputs[k]] = [grads[k]]


def find_topo_sort(nodes: List[Tensor]) -> List[Tensor]:
    topo_order = []
    for node in nodes:
        topo_sort_dfs(node, topo_order)
    return topo_order


def topo_sort_dfs(node: Tensor, topo_order: List[Tensor]):
    for input_node in node.inputs: topo_sort_dfs(input_node, topo_order)
    if node not in topo_order: topo_order.append(node)
