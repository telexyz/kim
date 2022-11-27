from typing import List, Optional, Tuple
import kim
import numpy as np

################################
####### Cấu trúc dữ liệu #######
####### trên Tensor      #######
################################

class TensorTupleOp:
    def __call__(self, *args):
        return TensorTuple.make_from_op(self, args)


class TensorTuple:
    # Data structure
    # `op` biến đổi input tensor(s) thành Tulple[Tensor] nên được gọi là TensorTuple
    # inputs có thể chỉ là 1 tensor khi op là split (tách tensor thành nhiều tensors con)
    op: Optional[TensorTupleOp]
    inputs: List["Tensor"]
    cached_data: Tuple["Tensor"]
    requires_grad: bool

    def realize_cached_data(self):
        if self.cached_data is None:
            self.cached_data = self.op.compute(*[x.realize_cached_data() for x in self.inputs])
        return self.cached_data

    def numpy(self):
        data = self.realize_cached_data()
        assert isinstance(data, tuple)
        if array_api is np:
            return list(data)
        else:
            return [x.numpy() for x in data]

    def assign_params_and_record_creation(
        self,
        op: Optional[TensorTupleOp]=None,
        inputs: List["Tensor"]=[],
        cached_data=None,
        requires_grad=False
    ):
        kim.autograd.CompGraph.NODE_COUNT += 1 # record new tensor creation
        self.op = op
        self.inputs = inputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad or any(tensor.requires_grad for tensor in inputs)

    def __len__(self):
        return len(self.realize_cached_data())

    # Truy cập vào 1 phần tử trong TensorTuple sử dụng qua ops.tuple_get_item
    # để còn tính được backward trong computational graph
    def __getitem__(self, index: int):
        return kim.ops.tuple_get_item(self, index)

    def tuple(self):
        n = len(self.realize_cached_data())
        return tuple([kim.ops.tuple_get_item(self, i) for i in range(n)])

    def __del__(self):
        kim.autograd.CompGraph.NODE_COUNT -= 1 # record tensor destruction

    def __repr__(self):
        return "kim.TensorTuple" + str(self.tuple())

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def make_from_op(op: TensorTupleOp, inputs: List["Tensor"]):
        tensor_tuple = TensorTuple.__new__(TensorTuple)
        tensor_tuple.assign_params_and_record_creation(op=op, inputs=inputs)
        if not kim.autograd.CompGraph.LAZY_MODE:
            if not tensor_tuple.requires_grad: return tensor_tuple.detach()
            tensor_tuple.realize_cached_data()
        return tensor_tuple

    def detach(self):
        return TensorTuple.make_const(self.realize_cached_data())

    @staticmethod
    def make_const(data, requires_grad=False) -> "TensorTuple":
        tensor_tuple = TensorTuple.__new__(TensorTuple)
        if isinstance(data, TensorTuple): data = data.realize_cached_data()
        tensor_tuple.assign_params_and_record_creation(
            op=None, 
            inputs=[],
            cached_data=data,
            requires_grad=requires_grad
        )
        return tensor_tuple

    # Hàm dùng để tính gradient (backward graph)
    # __add__ dùng trong sum gradient (autograd.py)
    def __add__(self, other):
        assert isinstance(other, TensorTuple)
        assert len(self) == len(other)
        return kim.ops.make_tensor_tuple(*[self[i] + other[i] for i in range(len(self))])
