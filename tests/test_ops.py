import numpy as np
import kim
import kim.nn as nn

from gradient_check import *
from kim import as_numpy

def get_tensor(*shape, entropy=1):
    np.random.seed(np.prod(shape) * len(shape) * entropy)
    return kim.Tensor(np.random.randint(0, 100, size=shape) / 20, dtype="float32")

def power_scalar_forward(shape, power=2):
    x = get_tensor(*shape)
    return (x**power).cached_data

def power_scalar_backward(shape, power=2):
    x = get_tensor(*shape)
    y = (x**power).sum()
    y.backward()
    return x.grad.cached_data

def logsumexp_forward(shape, axes):
    x = get_tensor(*shape)
    return (kim.ops.logsumexp(x,axes=axes)).cached_data

def logsumexp_backward(shape, axes):
    x = get_tensor(*shape)
    y = (kim.ops.logsumexp(x, axes=axes)**2).sum()
    y.backward()
    return x.grad.cached_data

def test_op_power_scalar_forward_1():
    np.testing.assert_allclose(as_numpy(power_scalar_forward((2,2), power=2)),
        np.array([[11.222499, 17.639997],
         [ 0.0625 , 20.25 ]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_op_power_scalar_forward_2():
    np.testing.assert_allclose(as_numpy(power_scalar_forward((2,2), power=-1.5)),
        np.array([[0.16309206, 0.11617859],
         [8. , 0.10475656]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_op_power_scalar_backward_1():
    np.testing.assert_allclose(as_numpy(power_scalar_backward((2,2), power=2)),
        np.array([[6.7, 8.4],
         [0.5, 9. ]], dtype=np.float32), rtol=1e-5, atol=1e-5)


def test_op_logsumexp_forward_1():
    np.testing.assert_allclose(logsumexp_forward((3,3,3),(1,2)),
        np.array([5.366029 , 4.9753823, 6.208126 ], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_op_logsumexp_forward_2():
    np.testing.assert_allclose(as_numpy(logsumexp_forward((3,3,3),None)),
        np.array([6.7517853], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_op_logsumexp_forward_3():
    np.testing.assert_allclose(logsumexp_forward((1,2,3,4),(0,2)),
        np.array([[5.276974 , 5.047317 , 3.778802 , 5.0103745],
       [5.087831 , 4.391712 , 5.025037 , 2.0214698]], dtype=np.float32), rtol=1e-5, atol=1e-5)


def test_op_logsumexp_forward_4():
    np.testing.assert_allclose(as_numpy(logsumexp_forward((3,10),(1,))),
        np.array([5.705309, 5.976375, 5.696459], dtype=np.float32), rtol=1e-5, atol=1e-5)


def test_op_logsumexp_forward_5():
    test_data = kim.ops.logsumexp(kim.Tensor(np.array([[1e10,1e9,1e8,-10],[1e-10,1e9,1e8,-10]])), (0,)).numpy()
    np.testing.assert_allclose(as_numpy(test_data), np.array([ 1.00000000e+10,  1.00000000e+09,  1.00000001e+08, -9.30685282e+00]), rtol=1e-5, atol=1e-5)

def test_op_logsumexp_backward_1():
    np.testing.assert_allclose(as_numpy(logsumexp_backward((3,1), (1,))),
        np.array([[1. ],
       [7.3],
       [9.9]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_op_logsumexp_backward_2():
    np.testing.assert_allclose(as_numpy(logsumexp_backward((3,3,3), (1,2))),
        np.array([[[1.4293308 , 1.2933122 , 0.82465225],
        [0.50017685, 2.1323113 , 2.1323113 ],
        [1.4293308 , 0.58112264, 0.40951014]],

       [[0.3578173 , 0.07983983, 4.359107  ],
        [1.1300558 , 0.561169  , 0.1132981 ],
        [0.9252113 , 0.65198547, 1.7722803 ]],

       [[0.2755132 , 2.365242  , 2.888913  ],
        [0.05291228, 1.1745441 , 0.02627547],
        [2.748018  , 0.13681579, 2.748018  ]]], dtype=np.float32), rtol=1e-5, atol=1e-5)


def test_op_logsumexp_backward_3():
    np.testing.assert_allclose(logsumexp_backward((3,3,3), (0,2)),
        np.array([[[0.92824626, 0.839912  , 0.5355515 ],
        [0.59857905, 2.551811  , 2.551811  ],
        [1.0213376 , 0.41524494, 0.29261813]],

       [[0.16957533, 0.03783737, 2.0658503 ],
        [0.98689   , 0.49007502, 0.09894446],
        [0.48244575, 0.3399738 , 0.9241446 ]],

       [[0.358991  , 3.081887  , 3.764224  ],
        [0.12704718, 2.820187  , 0.06308978],
        [3.9397335 , 0.19614778, 3.9397335 ]]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_op_logsumexp_backward_5():
    grad_compare = kim.Tensor(np.array([[1e10,1e9,1e8,-10],[1e-10,1e9,1e8,-10]]))
    test_data = (kim.ops.logsumexp(grad_compare, (0,))**2).sum().backward()
    np.testing.assert_allclose(as_numpy(grad_compare.grad.cached_data),
        np.array([[ 2.00000000e+10,  9.99999999e+08,  1.00000001e+08,
        -9.30685282e+00],
       [ 0.00000000e+00,  9.99999999e+08,  1.00000001e+08,
        -9.30685282e+00]]), rtol=1e-5, atol=1e-5)


def test_op_logsumexp_backward_4():
    np.testing.assert_allclose(as_numpy(logsumexp_backward((1,2,3,4), None)),
        np.array([[[[0.96463485, 1.30212122, 0.09671321, 1.84779774],
         [1.84779774, 0.39219132, 0.21523925, 0.30543892],
         [0.01952606, 0.55654611, 0.32109909, 0.01598658]],

        [[1.30212122, 0.83026929, 0.30543892, 0.01680623],
         [0.29054249, 0.07532032, 1.84779774, 0.05307731],
         [0.75125862, 0.26289377, 0.04802637, 0.03932065]]]], dtype=np.float32), rtol=1e-5, atol=1e-5)


# # # # # # # # # # # # # # # # # # # # # # # # # # # #

def test_divide_forward():
    np.testing.assert_allclose(kim.divide(kim.Tensor([[3.3 , 4.35, 1.2 ],
       [2.45, 0.95, 2.55]]), kim.Tensor([[4.6 , 4.35, 4.8 ],
       [0.65, 0.7 , 4.4 ]])).numpy(), np.array([[0.717391304348, 1., 0.25],
       [3.769230769231, 1.357142857143, 0.579545454545]]))


def test_divide_scalar_forward():
    np.testing.assert_allclose(kim.divide_scalar(
        kim.Tensor([[1.7 , 1.45]]), scalar=12).numpy(), 
        np.array([[0.141666666667, 0.120833333333]]))


def test_matmul_forward():
    np.testing.assert_allclose(kim.matmul(kim.Tensor([[4.95, 1.75, 0.25],
       [4.15, 4.25, 0.3 ],
       [0.3 , 0.4 , 2.1 ]]), kim.Tensor([[1.35, 2.2 , 1.55],
       [3.85, 4.8 , 2.6 ],
       [1.15, 0.85, 4.15]])).numpy(), np.array([[13.7075, 19.5025, 13.26],
       [22.31  , 29.785 , 18.7275],
       [ 4.36  ,  4.365 , 10.22  ]]))

    np.testing.assert_allclose(kim.matmul(kim.Tensor([[3.8 , 0.05],
       [2.3 , 3.35],
       [1.6 , 2.6 ]]), kim.Tensor([[1.1 , 3.5 , 3.7 ],
       [0.05, 1.25, 1.  ]])).numpy(), np.array([[ 4.1825, 13.3625, 14.11],
       [ 2.6975, 12.2375, 11.86  ],
       [ 1.89  ,  8.85  ,  8.52  ]]))

    np.testing.assert_allclose(kim.matmul(kim.Tensor([[[4.  , 2.15],
        [1.25, 1.35],
        [0.75, 1.6 ]],
       [[2.9 , 2.15],
        [3.3 , 4.1 ],
        [2.5 , 0.25]],
       [[2.9 , 4.35],
        [1.2 , 3.5 ],
        [3.55, 3.95]],
       [[2.55, 4.35],
        [4.25, 0.2 ],
        [3.95, 3.4 ]],
       [[2.2 , 2.05],
        [0.95, 1.8 ],
        [2.7 , 2.  ]],
       [[0.45, 1.1 ],
        [3.15, 0.7 ],
        [2.9 , 1.95]]]), kim.Tensor([[[2.7 , 4.05, 0.1 ],
        [1.75, 3.05, 2.3 ]],
       [[0.55, 4.1 , 2.3 ],
        [4.45, 2.35, 2.55]],
       [[1.2 , 3.95, 4.6 ],
        [4.2 , 3.5 , 3.35]],
       [[2.55, 4.4 , 2.05],
        [2.4 , 0.6 , 4.65]],
       [[2.95, 0.8 , 0.6 ],
        [0.45, 1.3 , 0.75]],
       [[1.25, 2.1 , 0.4 ],
        [0.85, 3.5 , 3.7 ]]])).numpy(), np.array([[[14.5625, 22.7575,  5.345 ],
        [ 5.7375,  9.18  ,  3.23  ],
        [ 4.825 ,  7.9175,  3.755 ]],
       [[11.1625, 16.9425, 12.1525],
        [20.06  , 23.165 , 18.045 ],
        [ 2.4875, 10.8375,  6.3875]],
       [[21.75  , 26.68  , 27.9125],
        [16.14  , 16.99  , 17.245 ],
        [20.85  , 27.8475, 29.5625]],
       [[16.9425, 13.83  , 25.455 ],
        [11.3175, 18.82  ,  9.6425],
        [18.2325, 19.42  , 23.9075]],
       [[ 7.4125,  4.425 ,  2.8575],
        [ 3.6125,  3.1   ,  1.92  ],
        [ 8.865 ,  4.76  ,  3.12  ]],
       [[ 1.4975,  4.795 ,  4.25  ],
        [ 4.5325,  9.065 ,  3.85  ],
        [ 5.2825, 12.915 ,  8.375 ]]]), rtol=1e-5, atol=1e-5)

    np.testing.assert_allclose(kim.matmul(kim.Tensor([[1.9 , 1.9 ],
       [4.8 , 4.9 ],
       [3.25, 3.75]]), kim.Tensor([[[1.25, 1.8 , 1.95],
        [3.75, 2.85, 2.25]],
       [[1.75, 2.7 , 3.3 ],
        [2.95, 1.55, 3.85]],
       [[4.2 , 3.05, 3.35],
        [3.3 , 4.75, 2.1 ]]])).numpy(), np.array([[[ 9.5   ,  8.835 ,  7.98  ],
        [24.375 , 22.605 , 20.385 ],
        [18.125 , 16.5375, 14.775 ]],
       [[ 8.93  ,  8.075 , 13.585 ],
        [22.855 , 20.555 , 34.705 ],
        [16.75  , 14.5875, 25.1625]],
       [[14.25  , 14.82  , 10.355 ],
        [36.33  , 37.915 , 26.37  ],
        [26.025 , 27.725 , 18.7625]]]), rtol=1e-5, atol=1e-5)

    np.testing.assert_allclose(kim.matmul(kim.Tensor([[[3.4 , 2.95],
        [0.25, 1.95],
        [4.4 , 4.4 ]],
       [[0.55, 1.1 ],
        [0.75, 1.55],
        [4.1 , 1.2 ]],
       [[1.5 , 4.05],
        [1.5 , 1.55],
        [2.3 , 1.25]]]), kim.Tensor([[2.2 , 0.65, 2.5 ],
       [2.5 , 1.3 , 0.15]])).numpy(), np.array([[[14.855 ,  6.045 ,  8.9425],
        [ 5.425 ,  2.6975,  0.9175],
        [20.68  ,  8.58  , 11.66  ]],
       [[ 3.96  ,  1.7875,  1.54  ],
        [ 5.525 ,  2.5025,  2.1075],
        [12.02  ,  4.225 , 10.43  ]],
       [[13.425 ,  6.24  ,  4.3575],
        [ 7.175 ,  2.99  ,  3.9825],
        [ 8.185 ,  3.12  ,  5.9375]]]))


def test_summation_forward():
    np.testing.assert_allclose(kim.summation(kim.Tensor([[2.2 , 4.35, 1.4 , 0.3 , 2.65],
       [1.  , 0.85, 2.75, 3.8 , 1.55],
       [3.2 , 2.3 , 3.45, 0.7 , 0.  ]])).numpy(), np.array(30.5))
    np.testing.assert_allclose(kim.summation(kim.Tensor([[1.05, 2.55, 1.  ],
       [2.95, 3.7 , 2.6 ],
       [0.1 , 4.1 , 3.3 ],
       [1.1 , 3.4 , 3.4 ],
       [1.8 , 4.55, 2.3 ]]), axes=1).numpy().flat, np.array([4.6 , 9.25, 7.5 , 7.9 , 8.65]))
    np.testing.assert_allclose(kim.summation(kim.Tensor([[1.5 , 3.85, 3.45],
       [1.35, 1.3 , 0.65],
       [2.6 , 4.55, 0.25]]), axes=0).numpy().flat, np.array([5.45, 9.7 , 4.35]))


def test_broadcast_to_forward():
    np.testing.assert_allclose(kim.broadcast_to(kim.Tensor([[1.85, 0.85, 0.6 ]]), shape=(3, 3, 3)).numpy(), np.array([[[1.85, 0.85, 0.6 ],
        [1.85, 0.85, 0.6 ],
        [1.85, 0.85, 0.6 ]],
       [[1.85, 0.85, 0.6 ],
        [1.85, 0.85, 0.6 ],
        [1.85, 0.85, 0.6 ]],
       [[1.85, 0.85, 0.6 ],
        [1.85, 0.85, 0.6 ],
        [1.85, 0.85, 0.6 ]]]))


def test_reshape_forward():
    np.testing.assert_allclose(kim.reshape(kim.Tensor([[2.9 , 2.  , 2.4 ],
       [3.95, 3.95, 4.65],
       [2.1 , 2.5 , 2.7 ],
       [1.9 , 4.85, 3.25],
       [3.35, 3.45, 3.45]]), shape=(15,)).numpy(), np.array([2.9 , 2.  , 2.4 , 3.95, 3.95, 4.65, 2.1 , 2.5 , 2.7 , 1.9 , 4.85,
       3.25, 3.35, 3.45, 3.45]))
    np.testing.assert_allclose(kim.reshape(kim.Tensor([[[4.1 , 4.05, 1.35, 1.65],
        [3.65, 0.9 , 0.65, 4.15]],
       [[4.7 , 1.4 , 2.55, 4.8 ],
        [2.8 , 1.75, 2.8 , 0.6 ]],
       [[3.75, 0.6 , 0.  , 3.5 ],
        [0.15, 1.9 , 4.75, 2.8 ]]]), shape=(2, 3, 4)).numpy(), np.array([[[4.1 , 4.05, 1.35, 1.65],
        [3.65, 0.9 , 0.65, 4.15],
        [4.7 , 1.4 , 2.55, 4.8 ]],
       [[2.8 , 1.75, 2.8 , 0.6 ],
        [3.75, 0.6 , 0.  , 3.5 ],
        [0.15, 1.9 , 4.75, 2.8 ]]]))

def test_negate_forward():
    np.testing.assert_allclose(kim.negate(kim.Tensor([[1.45, 0.55]])).numpy(), np.array([[-1.45, -0.55]]))


def test_transpose_forward():
    np.testing.assert_allclose(kim.transpose(kim.Tensor([[[1.95]],
       [[2.7 ]],
       [[3.75]]]), axes=(1, 2)).numpy(), np.array([[[1.95]],
       [[2.7 ]],
       [[3.75]]]))
    np.testing.assert_allclose(kim.transpose(kim.Tensor([[[[0.95]]],
       [[[2.55]]],
       [[[0.45]]]]), axes=(2, 3)).numpy(), np.array([[[[0.95]]],
       [[[2.55]]],
       [[[0.45]]]]))
    np.testing.assert_allclose(kim.transpose(kim.Tensor([[[[0.4 , 0.05],
         [2.95, 1.3 ]],
        [[4.8 , 1.2 ],
         [1.65, 3.1 ]]],
       [[[1.45, 3.05],
         [2.25, 0.1 ]],
        [[0.45, 4.75],
         [1.5 , 1.8 ]]],
       [[[1.5 , 4.65],
         [1.35, 2.7 ]],
        [[2.  , 1.65],
         [2.05, 1.2 ]]]])).numpy(), np.array([[[[0.4 , 2.95],
         [0.05, 1.3 ]],
        [[4.8 , 1.65],
         [1.2 , 3.1 ]]],
       [[[1.45, 2.25],
         [3.05, 0.1 ]],
        [[0.45, 1.5 ],
         [4.75, 1.8 ]]],
       [[[1.5 , 1.35],
         [4.65, 2.7 ]],
        [[2.  , 2.05],
         [1.65, 1.2 ]]]]))
    np.testing.assert_allclose(kim.transpose(kim.Tensor([[[2.45]],
       [[3.5 ]],
       [[0.9 ]]]), axes=(0, 1)).numpy(), np.array([[[2.45],
        [3.5 ],
        [0.9 ]]]))
    np.testing.assert_allclose(kim.transpose(kim.Tensor([[4.4 , 2.05],
       [1.85, 2.25],
       [0.15, 1.4 ]])).numpy(), np.array([[4.4 , 1.85, 0.15],
       [2.05, 2.25, 1.4 ]]))
    np.testing.assert_allclose(kim.transpose(kim.Tensor([[0.05, 3.7 , 1.35],
       [4.45, 3.25, 1.95],
       [2.45, 4.4 , 4.5 ]])).numpy(), np.array([[0.05, 4.45, 2.45],
       [3.7 , 3.25, 4.4 ],
       [1.35, 1.95, 4.5 ]]))
    np.testing.assert_allclose(kim.transpose(kim.Tensor([[[0.55, 1.8 , 0.2 ],
        [0.8 , 2.75, 3.7 ],
        [0.95, 1.4 , 0.8 ]],
       [[0.75, 1.6 , 1.35],
        [3.75, 4.  , 4.55],
        [1.85, 2.5 , 4.8 ]],
       [[0.2 , 3.35, 3.4 ],
        [0.3 , 4.85, 4.85],
        [4.35, 4.25, 3.05]]]), axes=(0, 1)).numpy(), np.array([[[0.55, 1.8 , 0.2 ],
        [0.75, 1.6 , 1.35],
        [0.2 , 3.35, 3.4 ]],
       [[0.8 , 2.75, 3.7 ],
        [3.75, 4.  , 4.55],
        [0.3 , 4.85, 4.85]],
       [[0.95, 1.4 , 0.8 ],
        [1.85, 2.5 , 4.8 ],
        [4.35, 4.25, 3.05]]]))

def test_log_forward():
    # test forward pass for log
    np.testing.assert_allclose(kim.log(kim.Tensor([[4.  ],
       [4.55]])).numpy(), np.array([[1.38629436112 ],
       [1.515127232963]]))

def test_relu_forward():
    np.testing.assert_allclose(kim.relu(kim.Tensor([[-46.9 , -48.8 , -45.45, -49.  ],
       [-49.75, -48.75, -45.8 , -49.25],
       [-45.65, -45.25, -49.3 , -47.65]])).numpy(), np.array([[0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]]))

# # # # # # # # # # # # # # # # # # # # # # # # # # # #

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
        scalar=np.random.randn(1)[0])


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

def test_summation_backward_multi_axis():
    gradient_check(kim.summation, kim.Tensor(np.random.randn(5,4)), axes=(0,1))
    gradient_check(kim.summation, kim.Tensor(np.random.randn(5,4,1)), axes=(0,1))


def test_backward_matmul_2d3d():
    np.random.seed(0)
    gradient_check(kim.divide, kim.Tensor(np.random.randn(3, 5)), 
        kim.Tensor(6 + np.random.randn(3, 5)))

    gradient_check(kim.divide_scalar, kim.Tensor(np.random.randn(3, 5)), 
        scalar=np.random.randn(1))

    gradient_check(kim.matmul, kim.Tensor(np.random.randn(1, 5)), 
        kim.Tensor(np.random.randn(5, 1)))

    gradient_check(kim.matmul, kim.Tensor(np.random.randn(2, 4)), 
        kim.Tensor(np.random.randn(4, 2)))

    gradient_check(kim.matmul, kim.Tensor(np.random.randn(7, 4, 2)), kim.Tensor(np.random.randn(2, 4)))

    gradient_check(kim.matmul, kim.Tensor(np.random.randn(2, 4)), 
        kim.Tensor(np.random.randn(7, 4, 2)))

def test_backward_matmul():
    gradient_check(kim.matmul, kim.Tensor(np.random.randn(3, 2, 1)), 
        kim.Tensor(np.random.randn(3, 3, 1, 2)))

    gradient_check(kim.matmul, kim.Tensor(np.random.randn(2, 4)), 
        kim.Tensor(np.random.randn(2, 4, 4, 2)))


def test_backward():
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

    gradient_check(kim.tanh, kim.Tensor(np.random.randn(5,4,5)))


def test_log_backward():
    gradient_check(kim.log, kim.Tensor(1 + np.random.rand(5,4)))


def test_relu_backward():
    gradient_check(kim.relu, kim.Tensor(np.random.randn(5,4)))
