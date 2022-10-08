import kim
from typing import List, Optional, NamedTuple, Tuple, Union
from collections import namedtuple

import numpy
# Tạm thời dùng numpy làm backend
import numpy as array_api
NDArray = numpy.ndarray

class State:
	LAZY_MODE = False
	TENSOR_COUNTER = 0

class Device:
	"""Thiết bị hỗ trợ NDArray."""

class CPUDevice(Device):
    """Dữ liệu nằm trên CPU"""

    # “official” string representation of an object, typically used for debugging.
    def __repr__(self):
        return "kim.cpu()"

    def __hash__(self): # Để dùng cho __eq__
        return self.__repr__().__hash__()

    def __eq__(self, other):
        return isinstance(other, CPUDevice)

    def enabled(self):
        return True

def cpu():
    """Return cpu device"""
    return CPUDevice()

def all_devices():
    """return a list of all available devices"""
    return [cpu()]


class TensorOp:
    """Định nghĩa toán tử thao tác trên Tensor"""
    def __call__(self, *args):
        return Tensor.make_from_op(self, args)

    def compute(self, *args: Tuple[NDArray]):
        """Calculate forward pass of operator.

        Parameters
        ----------
        input: Tuple[NDArray]
            A list of input arrays to the function

        Returns
        -------
        output: NDArray
            Array output of the operation
        """
        raise NotImplementedError()

    def gradient(self, out_grad: "Tensor", node: "Tensor"
    	) -> Union["Tensor", Tuple["Tensor"]]:
        """Compute partial adjoint for each input value for a given output adjoint.

        Parameters
        ----------
        out_grad: Tensor
            The adjoint wrt to the output value.

        node: Tensor
            The value node of forward evaluation.

        Returns
        -------
        input_gradients: Tensor hoặc Tuple[Tensor]
            Trả về gradient liền kề từng phần tương ứng với mỗi đầu vào của `node`.
        """
        raise NotImplementedError()

    def gradient_as_tuple(self, out_grad: "Tensor", node: "Tensor") -> Tuple["Tensor"]:
        """Luôn trả về gradient dưới dạng tupple để tiện cho việc tính toán.
        Lý do: mỗi op có 1 hoặc 2 input tensors. 
        	- Với 2 inputs trả về tupple (out1, out2)
        	- Với 1 input  trả về tupple (out1, )
        """
        output = self.gradient(out_grad, node)
        if isinstance(output, tuple):
            return output
        elif isinstance(output, list):
            return tuple(output) # biến list thành tupple
        else:
            return (output,) # tupple chỉ có 1 phần tử


class Tensor:
    """Một nốt trong đồ thị tính toán. Đây là cấu trúc dữ liệu quan trọng nhất
    để tạo nên đồ thị tính toán bao gồm:

    - Khởi tạo đồ thị từ Tensor's `inputs`
    - Tính toán forward, dùng `op.compute()`
    - Tính toán backward, dùng `op.gradient()`
    """
    grad: "Tensor" # gradient của node được lưu lại khi gọi backward

    op: Optional[TensorOp] # toán tử để tính ra Tensor hiện tại
    inputs: List["Tensor"]
    """
    Mỗi nốt lưu `inputs` là các Tensor đầu vào của `op`
    `op` biến đổi `inputs` thành giá trị đầu ra của nốt và
    lưu trữ tại `cached_data`. 
    
    Khi dùng đến giá trị đầu ra ta gọi hàm `tensor.data()`

    Hàm này gọi `tensor.realize_cached_data()`, nếu `op` đã được thực hiện 
    thì trả về kết quả lưu trong `cached_data`. Còn không thì ta thực hiện
    tính toán và lưu kết quả trong `cached_data` rồi trả về. Lưu ý:

    - Với Tensor không có `op` và `inputs` thì gán thẳng giá trị đầu vào cho `cached_data`

    - Đầu vào của hàm `op.compute()` là NDArray, không phải Tensor
    """

    # The following fields are cached fields for dynamic computation
    cached_data: NDArray
    requires_grad: bool

    def realize_cached_data(self):
        """Run compute to realize the cached data"""
        if self.cached_data is None:
	        self.cached_data = self.op.compute(
	            *[x.realize_cached_data() for x in self.inputs]
            )
        return self.cached_data


    def is_leaf(self):
        '''Tensor là lá khi không có op'''
        return self.op is None


    def __del__(self):
        State.TENSOR_COUNTER -= 1


    def assign_params_n_record_creation(
        self, op: Optional[TensorOp],
        inputs: List["Tensor"], *,
    	cached_data: List[object] = None,
        requires_grad: Optional[bool] = None
    ):
        '''Hàm này sẽ chính thức ghi nhận việc khởi tạo Tensor bằng cách
        gán các tham số cần thiết và tăng TENSOR_COUNTER lên 1.
        Vì thế nó sẽ luôn được gọi sau khi mọt tensor mới được tạo ra.
        '''
        State.TENSOR_COUNTER += 1
        
        # Tự động tìm xem có cần tính gradient với không dựa và inputs
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)

        self.op = op
        self.inputs = inputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad


    @staticmethod
    def make_const(data, *, requires_grad=False):
        tensor = Tensor(None)
        tensor.assign_params_n_record_creation(
            op=None, inputs=[],
            cached_data=data,
            requires_grad=requires_grad,
        )
        return tensor

    def __init__(self, array, *,
        device: Optional[Device] = None,
        dtype=None,
        requires_grad=True,
        **kwargs
    ):
        if isinstance(array, Tensor):
            if device is None:
                device = array.device

            if dtype is None:
                dtype = array.dtype

            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()

            else: # fall back, copy through numpy conversion
                cached_data = get_array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        else:
            device = device if device else cpu()
            cached_data = get_array_from_numpy(array, device=device, dtype=dtype)

        self.assign_params_n_record_creation(
            op=None, inputs=[],
            cached_data=cached_data,
            requires_grad=requires_grad,
        )

    '''Static method, giống như các class method, là những method được liên kết với một class chứ không phải đối tượng của nó.
    '''
    @staticmethod
    def make_from_op(op: TensorOp, inputs: List["Tensor"]):
        # tensor = Tensor(None) # Tạo một tensor mới
        tensor = Tensor.__new__(Tensor) # Tạo một tensor mới
        tensor.assign_params_n_record_creation(op=op, inputs=inputs)
        if not State.LAZY_MODE:
            if not tensor.requires_grad:
                return tensor.detach()
            tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data, requires_grad=False):
        # tensor = Tensor(None) # Tạo một tensor mới
        tensor = Tensor.__new__(Tensor) # Tạo một tensor mới
        tensor.assign_params_n_record_creation(
            op=None,
            inputs=[],
            cached_data=data
            if not isinstance(data, Tensor)
            else data.realize_cached_data(),
            requires_grad=requires_grad,
        )
        return tensor


    @property
    def data(self):
        return self.detach()

    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, "%s %s" % (
            value.dtype,
            self.dtype,
        )
        self.cached_data = value.realize_cached_data()

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return Tensor.make_const(self.realize_cached_data())

    @property
    def shape(self):
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        return self.realize_cached_data().dtype

    @property
    def device(self):
        # numpy array always sits on cpu
        if array_api is numpy:
            return cpu()
        else:
            return self.realize_cached_data().device


    def backward(self, out_grad=None):
        if out_grad is None: out_grad = Tensor(numpy.ones(self.shape))
        compute_gradient_of_backward_graph_variables(self, out_grad)


    def __repr__(self):
        return "kim.Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        return self.realize_cached_data().__str__()


    def numpy(self):
        data = self.realize_cached_data()
        if array_api is numpy:
            return data
        else:
            return data.numpy()


    def __add__(self, other):
        if isinstance(other, Tensor):
            return kim.ops.EWiseAdd()(self, other)
        else:
            return kim.ops.AddScalar(other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return kim.ops.EWiseMul()(self, other)
        else:
            return kim.ops.MulScalar(other)(self)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            raise NotImplementedError()
        else:
            return kim.ops.PowerScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return kim.ops.EWiseAdd()(self, kim.ops.Negate()(other))
        else:
            return kim.ops.AddScalar(-other)(self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return kim.ops.EWiseDiv()(self, other)
        else:
            return kim.ops.DivScalar(other)(self)

    def __matmul__(self, other):
        return kim.ops.MatMul()(self, other)

    def matmul(self, other):
        return kim.ops.MatMul()(self, other)

    def sum(self, axes=None):
        return kim.ops.Summation(axes)(self)

    def broadcast_to(self, shape):
        return kim.ops.BroadcastTo(shape)(self)

    def reshape(self, shape):
        return kim.ops.Reshape(shape)(self)

    def __neg__(self):
        return kim.ops.Negate()(self)

    def transpose(self, axes=None):
        return kim.ops.Transpose(axes)(self)

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    __rmatmul__ = __matmul__


##############################
####### Helper Methods #######
##############################

'''Các hàm tiện ích khác
'''
def get_array_from_numpy(numpy_array, device, dtype):
    if array_api is numpy:
        return numpy.array(numpy_array, dtype=dtype)
    return array_api.array(numpy_array, device=device, dtype=dtype)


def compute_gradient_of_backward_graph_variables(output_tensor, out_grad):
    """Đầu vào: gradient (out_grad) của một output nốt (output_tensor)
    Đầu ra: lưu trữ gradient ngược (backward) vào các trường `grad` 
    của các biến tương ứng trong đồ thị tính toán (các biến ở đây chính là 
    các tensors).

    Quá trình này sẽ tạo ra đồ thị tính toán ngược cho backward bắt đầu từ 
    `output_tensor` node. Bằng việc gọi hàm `gradient()` của từng op của node theo
    thứ tự topo ngược của đồ thị tính toán xuôi, và các toán tử trong `gradient()`
    tạo ra các node mới liên kết `out_grad` và các `node.inputs` để tạo nên `.grad`
    của từng node của cả đồ thị tính toán ngược và xuôi.
    """

    # Ánh xạ tới các đóng góp gradient của một nốt trong trong đồ thị tính toán
    # Một node có thể là đầu vào (toán hạng) của nhiều toán tử nên backward gradient 
    # nó là tổng gradient được đóng góp từ các ops có sử dụng nó.
    output_grads: Dict[Tensor, List[Tensor]] = {}

    # Lưu ý đặc biệt về khởi tạo gradient
    # Chúng ta thực sự đang lấy một đạo hàm của vô hướng reduce_sum(output_node)
    # thay vì vector output_node. Đây là trường hợp phổ biến đối với hàm mất mát 
    output_grads[output_tensor] = [out_grad]

    # Duyệt đồ thị tính toán theo chiều ngược, bắt đầu từ `output_tensor` để tính
    # gradient lan truyền ngược trở lại các input nodes. Trong quá trình đó,
    # đồ thị tính toán ngược sẽ được hình thành từ `out_grad` và `node.inputs`
    # của các node tương ứng được duyệt
    reverse_topo_order = reversed(find_topo_sort([output_tensor]))

    for node in reverse_topo_order:
        out_grads = output_grads[node]
        # print(">>>", len(out_grads), out_grads)

        node.grad = out_grads[0]
        for i in range(len(out_grads) - 1):
            node.grad = kim.add(node.grad, out_grads[i + 1])
        # print(">>>", node.grad)

        if node.op:
            input_grads = node.op.gradient_as_tuple(node.grad, node)

            for k in range(len(node.inputs)):
                input_k_tensor = node.inputs[k]
                input_k_grad = input_grads[k]
                try:
                    output_grads[input_k_tensor].append( input_k_grad )
                except KeyError:
                    # nếu output_grads của `input_tensor` chưa được
                    # khởi tạo thì khởi tạo lần đầu
                    output_grads[input_k_tensor] = [ input_k_grad ]


def find_topo_sort(node_list: List[Tensor]) -> List[Tensor]:
    """
    Cho một danh sách các nodes, trả lại danh sách topological sort của các nodes kết thúc tại danh sách các nodes đó.

    Một thuật toán đơn giản là thực hiện post-order DFS travel với các nodes đầu vào, 
    đi ngược lại dựa trên các cạnh đầu vào. Vì các node đưọc thêm vào sau khi các 
    predecessors được đi tới, chúng ta có được sắp xếp topo hình học.
    """
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, topo_order)
    return topo_order


def topo_sort_dfs(node, topo_order):
    """Post-order DFS"""
    for n in node.inputs:
        topo_sort_dfs(n, topo_order)

    if node not in topo_order:
        topo_order.append(node)
