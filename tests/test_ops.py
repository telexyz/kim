import numpy as np
import kim
import kim.nn as nn

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
    np.testing.assert_allclose(power_scalar_forward((2,2), power=2),
        np.array([[11.222499, 17.639997],
         [ 0.0625 , 20.25 ]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_op_power_scalar_forward_2():
    np.testing.assert_allclose(power_scalar_forward((2,2), power=-1.5),
        np.array([[0.16309206, 0.11617859],
         [8. , 0.10475656]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_op_power_scalar_backward_1():
    np.testing.assert_allclose(power_scalar_backward((2,2), power=2),
        np.array([[6.7, 8.4],
         [0.5, 9. ]], dtype=np.float32), rtol=1e-5, atol=1e-5)


def test_op_logsumexp_forward_1():
    np.testing.assert_allclose(logsumexp_forward((3,3,3),(1,2)),
        np.array([5.366029 , 4.9753823, 6.208126 ], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_op_logsumexp_forward_2():
    np.testing.assert_allclose(logsumexp_forward((3,3,3),None),
        np.array([6.7517853], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_op_logsumexp_forward_3():
    np.testing.assert_allclose(logsumexp_forward((1,2,3,4),(0,2)),
        np.array([[5.276974 , 5.047317 , 3.778802 , 5.0103745],
       [5.087831 , 4.391712 , 5.025037 , 2.0214698]], dtype=np.float32), rtol=1e-5, atol=1e-5)


def test_op_logsumexp_forward_4():
    np.testing.assert_allclose(logsumexp_forward((3,10),(1,)),
        np.array([5.705309, 5.976375, 5.696459], dtype=np.float32), rtol=1e-5, atol=1e-5)


def test_op_logsumexp_forward_5():
    test_data = kim.ops.logsumexp(kim.Tensor(np.array([[1e10,1e9,1e8,-10],[1e-10,1e9,1e8,-10]])), (0,)).numpy()
    np.testing.assert_allclose(test_data,np.array([ 1.00000000e+10,  1.00000000e+09,  1.00000001e+08, -9.30685282e+00]), rtol=1e-5, atol=1e-5)

def test_op_logsumexp_backward_1():
    np.testing.assert_allclose(logsumexp_backward((3,1), (1,)),
        np.array([[1. ],
       [7.3],
       [9.9]], dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_op_logsumexp_backward_2():
    np.testing.assert_allclose(logsumexp_backward((3,3,3), (1,2)),
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
    np.testing.assert_allclose(grad_compare.grad.cached_data,np.array([[ 2.00000000e+10,  9.99999999e+08,  1.00000001e+08,
        -9.30685282e+00],
       [ 0.00000000e+00,  9.99999999e+08,  1.00000001e+08,
        -9.30685282e+00]]), rtol=1e-5, atol=1e-5)


def test_op_logsumexp_backward_4():
    np.testing.assert_allclose(logsumexp_backward((1,2,3,4), None),
        np.array([[[[0.96463485, 1.30212122, 0.09671321, 1.84779774],
         [1.84779774, 0.39219132, 0.21523925, 0.30543892],
         [0.01952606, 0.55654611, 0.32109909, 0.01598658]],

        [[1.30212122, 0.83026929, 0.30543892, 0.01680623],
         [0.29054249, 0.07532032, 1.84779774, 0.05307731],
         [0.75125862, 0.26289377, 0.04802637, 0.03932065]]]], dtype=np.float32), rtol=1e-5, atol=1e-5)
