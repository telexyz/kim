import kim
from typing import List, Optional, NamedTuple, Tuple, Union

# Dùng numpy làm array_api backend
import numpy as array_api
import numpy
NDArray = numpy.ndarray

class State:
    LAZY_MODE = False
    TENSOR_COUNT = 0

############################################
####### Devices: phần cứng tính toán #######
############################################

class Device:
    """Phần cứng tính toán trên dữ liệu NDArray. Có thể là CPU, GPU hoặc TPU"""

class CpuDevice(Device):
    def __repr__(self):
        return "kim.cpu()"
    def enabled(self):
        return True

def cpu():
    return CpuDevice() # một instant mới của class CpuDevice

def all_devices():
    return [cpu()]

##################################
####### Tensor và TensorOp #######
##################################

class TensorOp:
    def __call__(self, *args):
        return Tensor.make_from_op(self, args)
    def compute():
        raise NotImplementedError()
    def gradient(self, out_grad: "Tensor", node: "Tensor") -> Tuple["Tensor"]:
        raise NotImplementedError()

class Tensor:
    def __repr__(self):
        return "kim.Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        return self.realize_cached_data().__str__()

    def __del__(self):
        State.TENSOR_COUNT -= 1
    
    grad: "Tensor"
    # Optional[...] is a shorthand notation for Union[..., None]
    cached_data: Optional[NDArray]
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
                cached_data = array.realize_cache_data()
        else:
            if device is None: device = cpu()
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
            if not tensor.requires_grad: tensor.detach() # tách khỏi đồ thị tính toán 
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
    
    
    def detached(self):
        return Tensor.make_const(self.realize_cached_data())

    @property
    def data(self): return self.detached()    

    @property
    def shape(self): return self.realize_cached_data().shape

    @property
    def dtype(self): return self.realize_cached_data().dtype

    @property
    def device(self):
        if array_api is numpy: return cpu()
        else: return self.realize_cached_data().device

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

    def sum(self, axes=None):
        return kim.ops.Summation(axes)(self)

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__

    def backward(self, out_grad=Optional["Tensor"]):
        if out_grad is None: out_grad = Tensor(numpy.ones(self.shape))
        compute_gradient_of_backward_graph_variables(self, out_grad)

##############################
####### Helper Methods #######
##############################

def get_array_from_numpy(numpy_array, device, dtype):
    if array_api is numpy: return numpy.array(numpy_array, dtype=dtype)
    else: return array_api.array(numpy_array, device=device, dtype=dtype)


def compute_gradient_of_backward_graph_variables(output_tensor, out_grad):
    output_grads: Dict[Tensor, List[Tensor]] = {}
    output_grads[output_tensor] = [out_grad]
    reverse_topo_order = reversed(find_topo_sort([output_tensor]))

    for node in reverse_topo_order:
        # tính node.grad
        out_grads = output_grads[node]
        node.grad = out_grads[0]
        for i in range(len(out_grads) - 1):
            node.grad = kim.add(node.grad, out_grads[i + 1])
        # 
        if node.op:
            input_grads = node.op.gradient(node.grad, node)
            for k in range(len(node.inputs)):
                input_k_tensor = node.inputs[k]
                input_k_grad = input_grads[k]
                try: output_grads[input_k_tensor].append(input_k_grad)
                except KeyError: output_grads[input_k_tensor] = [input_k_grad]


def find_topo_sort(nodes: List[Tensor]) -> List[Tensor]:
    topo_order = []
    for node in nodes:
        topo_sort_dfs(node, topo_order)
    return topo_order

def topo_sort_dfs(node, topo_order):
    for input_node in node.inputs: topo_sort_dfs(input_node, topo_order)
    if node not in topo_order: topo_order.append(node)
