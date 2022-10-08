import numpy as np
import numdifftools as nd
import kim

from gradient_check import *

def test_add_backward():
    gradient_check(kim.add, kim.Tensor(np.random.randn(5, 4)),
        kim.Tensor(5 + np.random.randn(5, 4)))


def test_add_scalar_backward():
    gradient_check(kim.add_scalar, kim.Tensor(np.random.randn(5, 4)), scalar=3)


def test_multiply_backward():
    gradient_check(kim.multiply, kim.Tensor(np.random.randn(5, 4)),
        kim.Tensor(5 + np.random.randn(5, 4)))


def test_mul_scalar_backward():
    gradient_check(kim.mul_scalar, kim.Tensor(np.random.randn(5, 4)), scalar=3)


def test_power_scalar_backward():
    gradient_check(kim.power_scalar, kim.Tensor(np.random.randn(5, 4)), scalar=3)


def test_divide_backward():
    gradient_check(kim.divide, kim.Tensor(np.random.randn(5, 4)),
        kim.Tensor(5 + np.random.randn(5, 4)))


def test_divide_scalar_backward():
    gradient_check(kim.divide_scalar, kim.Tensor(np.random.randn(5, 4)),
        scalar=np.random.randn(1))


def test_matmul_simple_backward():
    gradient_check(kim.matmul, kim.Tensor(np.random.randn(5, 4)), kim.Tensor(np.random.randn(4, 5)))


def test_matmul_batched_backward():
    gradient_check(kim.matmul, kim.Tensor(np.random.randn(6, 6, 5, 4)), kim.Tensor(np.random.randn(6, 6, 4, 3)))
    gradient_check(kim.matmul, kim.Tensor(np.random.randn(6, 6, 5, 4)), kim.Tensor(np.random.randn(4, 3)))
    gradient_check(kim.matmul, kim.Tensor(np.random.randn(5, 4)), kim.Tensor(np.random.randn(6, 6, 4, 3)))


def test_reshape_backward():
    gradient_check(kim.reshape, kim.Tensor(np.random.randn(5, 4)), shape=(4, 5))
    gradient_check(kim.reshape, kim.Tensor(np.random.randn(5, 4)), shape=(5, 4, 1))
    gradient_check(kim.reshape, kim.Tensor(np.random.randn(5, 4)), shape=(2, 2, 5))

def test_negate_backward():
    gradient_check(kim.negate, kim.Tensor(np.random.randn(5, 4)))


def test_transpose_backward():
    gradient_check(kim.transpose, kim.Tensor(np.random.randn(3, 5, 4)), axes=(1, 2))
    gradient_check(kim.transpose, kim.Tensor(np.random.randn(3, 5, 4)), axes=(0, 1))

def test_broadcast_to_backward():
    gradient_check(kim.broadcast_to, kim.Tensor(np.random.randn(3, 1)), shape=(3, 3))
    gradient_check(kim.broadcast_to, kim.Tensor(np.random.randn(1, 3)), shape=(3, 3))
    gradient_check(kim.broadcast_to, kim.Tensor(np.random.randn(1,)), shape=(3, 3, 3))
    gradient_check(kim.broadcast_to, kim.Tensor(np.random.randn()), shape=(3, 3, 3))
    gradient_check(kim.broadcast_to, kim.Tensor(np.random.randn(5,4,1)), shape=(5,4,3))


def test_summation_backward():
    gradient_check(kim.summation, kim.Tensor(np.random.randn(5,4)), axes=(1,))
    gradient_check(kim.summation, kim.Tensor(np.random.randn(5,4)), axes=(0,))
    gradient_check(kim.summation, kim.Tensor(np.random.randn(5,4)), axes=(0,1))
    gradient_check(kim.summation, kim.Tensor(np.random.randn(5,4,1)), axes=(0,1))


def test_backward():
    np.random.seed(0)
    gradient_check(kim.divide, kim.Tensor(np.random.randn(3, 5)), 
        kim.Tensor(6 + np.random.randn(3, 5)))

    gradient_check(kim.divide_scalar, kim.Tensor(np.random.randn(3, 5)), 
        scalar=np.random.randn(1))
    gradient_check(kim.matmul, kim.Tensor(np.random.randn(1, 5)), 
        kim.Tensor(np.random.randn(5, 1)))

    gradient_check(kim.matmul, kim.Tensor(np.random.randn(2, 4)), 
        kim.Tensor(np.random.randn(4, 2)))

    gradient_check(kim.matmul, kim.Tensor(np.random.randn(2, 4)), 
        kim.Tensor(np.random.randn(7, 4, 2)))

    gradient_check(kim.matmul, kim.Tensor(np.random.randn(3, 2, 1)), 
        kim.Tensor(np.random.randn(3, 3, 1, 2)))

    gradient_check(kim.matmul, kim.Tensor(np.random.randn(2, 4)), 
        kim.Tensor(np.random.randn(2, 4, 4, 2)))

    gradient_check(kim.reshape, kim.Tensor(np.random.randn(5, 4)), shape=(5,4,1))
    gradient_check(kim.reshape, kim.Tensor(np.random.randn(5, 4)), shape=(2, 2, 5))
    gradient_check(kim.negate, kim.Tensor(np.random.randn(1, 4, 2)))
    gradient_check(kim.transpose, kim.Tensor(np.random.randn(3, 2, 4)), axes=(0, 2))
    gradient_check(kim.broadcast_to, kim.Tensor(np.random.randn(7, 1)), shape=(7, 7))
    gradient_check(kim.broadcast_to, kim.Tensor(np.random.randn(1, 5)), shape=(5, 5))
    gradient_check(kim.broadcast_to, kim.Tensor(np.random.randn(1,)), shape=(4, 4, 4))
    gradient_check(kim.broadcast_to, kim.Tensor(np.random.randn()), shape=(1, 3, 6))
    gradient_check(kim.broadcast_to, kim.Tensor(np.random.randn(4,4,1)), shape=(4,4,6))
    gradient_check(kim.summation, kim.Tensor(np.random.randn(3,2,1)))
    gradient_check(kim.summation, kim.Tensor(np.random.randn(3,6)), axes=(1,))
    gradient_check(kim.summation, kim.Tensor(np.random.randn(7,)), axes=(0,))
    gradient_check(kim.summation, kim.Tensor(np.random.randn(7,8)), axes=(0,1))
    gradient_check(kim.summation, kim.Tensor(np.random.randn(5,4,5)), axes=(0,1,2))


def test_log_backward():
    gradient_check(kim.log, kim.Tensor(1 + np.random.rand(5,4)))


def test_relu_backward():
    gradient_check(kim.relu, kim.Tensor(np.random.randn(5,4)))
