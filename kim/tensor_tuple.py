from typing import List, Optional, Union
import kim

################################
####### Cấu trúc dữ liệu #######
####### trên Tensor      #######
################################

class TensorTupleOp:
    def __call__(self, *args):
        return TensorTuple.make_from_op(self, args)


class TensorTuple:
    op: Optional[TensorTupleOp]
    inputs: List["TensorTuple"]
    cached_data: Optional["TensorTuple"]
    requires_grad: bool

    def realize_cached_data(self):
        if self.cached_data is None:
            self.cached_data = self.op.compute(
                *[x.realize_cached_data() for x in self.inputs]
            )
        return self.cached_data

    def numpy(self):
        data = self.realize_cached_data()
        if array_api is numpy: return data
        if isinstance(data, tuple):
            return [x.numpy() for x in data]
        else:
            return data.numpy()

    def assign_params_and_record_creation(
        self, op: Optional[TensorTupleOp],
        inputs: List["TensorTuple"],
        cached_data = None,
        requires_grad: bool = False
    ):
        kim.State.TENSOR_COUNT += 1
        if not requires_grad:
            requires_grad = any(x.requires_grad for x in inputs)
        self.op = op
        self.inputs = inputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad

    def __len__(self):
        return len(self.realize_cached_data())

    def __getitem__(self, index: int):
        return kim.ops.tuple_get_item(self, index)

    def tuple(self):
        return tuple([x for x in self])

    def __del__(self):
        kim.State.TENSOR_COUNT -= 1

    def __repr__(self):
        return "kim.TensorTuple" + str(self.tuple())

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        assert isinstance(other, TensorTuple)
        assert len(self) == len(other)
        return kim.ops.make_tuple(*[self[i] + other[i] for i in range(len(self))])

    __radd__ = __add__

    def detach(self):
        return TensorTuple.make_const(self.realize_cached_data())

    @staticmethod
    def make_from_op(op: TensorTupleOp, inputs: List["TensorTuple"]):
        tensor_tuple = TensorTuple.__new__(TensorTuple)
        tensor_tuple.assign_params_and_record_creation(op=op, inputs=inputs)
        if not kim.State.LAZY_MODE:
            if not tensor_tuple.requires_grad:
                return tensor_tuple.detach()
            tensor_tuple.realize_cached_data()
        return tensor_tuple

    @staticmethod
    def make_const(data, requires_grad=False) -> "TensorTuple":
        tensor_tuple = TensorTuple.__new__(TensorTuple)
        if isinstance(data, TensorTuple): data = data.realize_cached_data()
        tensor_tuple.assign_params_and_record_creation(
            op=None, inputs=[],
            cached_data=data,
            requires_grad=requires_grad
        )
        return tensor_tuple
