import itertools
import numpy as np
import pytest
import mugrade
import torch

import kim
from kim import backend_ndarray as nd
from backend_select import CPU_CUDA, _DEVICES


np.random.seed(1)

def backward_check(f, *args, **kwargs):
    eps = 1e-5
    out = f(*args, **kwargs)
    c = np.random.randn(*out.shape)
    numerical_grad = [np.zeros(a.shape) for a in args]
    num_args = len(args)
    for i in range(num_args):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            f1 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] -= 2 * eps
            f2 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] += eps
            numerical_grad[i].flat[j] = (f1 - f2) / (2 * eps)
    backward_grad = out.op.gradient(kim.Tensor(c, device=args[0].device), out)
    error = sum(
        np.linalg.norm(backward_grad[i].numpy() - numerical_grad[i])
        for i in range(len(args))
    )
    assert error < 4.2e-1 # original is 4.2e-1
    return [g.numpy() for g in backward_grad]



EWISE_OPS = {
    "divide": lambda a, b: a / b,
    "subtract": lambda a, b: a - b
}
EWISE_OP_FNS = [EWISE_OPS[k] for k in EWISE_OPS]
EWISE_OP_NAMES = [k for k in EWISE_OPS]
GENERAL_SHAPES = [(1, 1, 1), (4, 5, 6), (3, 1, 2), (1, 1), (8, 15, 1)]
@pytest.mark.parametrize("fn", EWISE_OP_FNS, ids=EWISE_OP_NAMES)
@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=CPU_CUDA)
def test_ewise_fn(fn, shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    _B = np.random.randn(*shape).astype(np.float32)
    A = kim.Tensor(nd.array(_A), device=device)
    B = kim.Tensor(nd.array(_B), device=device)
    fn_A_B = fn(_A, _B)
    fnAB = fn(A, B).numpy()
    # print(">>>", fn_A_B, fnAB)
    # print(">>>", fn_A_B.dtype, fnAB.dtype)
    # print(">>>", type(fn_A_B), type(fnAB))
    np.testing.assert_allclose(fn_A_B, fnAB, atol=1e-5, rtol=1e-5)


SCALAR_OPS = {
    "divide": lambda a, b: a / b,
    "subtract": lambda a, b: a - b
}
SCALAR_OP_FNS = [SCALAR_OPS[k] for k in SCALAR_OPS]
SCALAR_OP_NAMES = [k for k in SCALAR_OPS]
@pytest.mark.parametrize("fn", SCALAR_OP_FNS, ids=SCALAR_OP_NAMES)
@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=CPU_CUDA)
def test_scalar_fn(fn, shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    _B = np.random.randn(1).astype(np.float32).item()
    A = kim.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(fn(_A, _B), fn(A, _B).numpy(), atol=1e-5, rtol=1e-5)


MATMUL_DIMS = [(16, 16, 16),
    (8, 8, 8),
    (1, 2, 3),
    (3, 4, 5),
    (5, 4, 3),
    (16, 16, 32),
    (64, 64, 64),
    (72, 72, 72),
    (72, 73, 74),
    (74, 73, 72),
    (128, 128, 128)]
@pytest.mark.parametrize("m,n,p", MATMUL_DIMS)
@pytest.mark.parametrize("device", _DEVICES, ids=CPU_CUDA)
def test_matmul(m, n, p, device):
    _A = np.random.randn(m, n).astype(np.float32)
    _B = np.random.randn(n, p).astype(np.float32)
    A = kim.Tensor(nd.array(_A), device=device)
    B = kim.Tensor(nd.array(_B), device=device)
    np.testing.assert_allclose(_A @ _B, (A @ B).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("m,n,p", MATMUL_DIMS)
@pytest.mark.parametrize("device", _DEVICES, ids=CPU_CUDA)
def test_batch_matmul(m, n, p, device):
    b = 6
    _A = np.random.randn(m, n).astype(np.float32)
    _B = np.random.randn(n, p).astype(np.float32)
    A = kim.Tensor(nd.array(_A), device=device)
    B = kim.Tensor(nd.array(_B), device=device)

    _Ab = np.random.randn(b, m, n).astype(np.float32)
    _Bb = np.random.randn(b, n, p).astype(np.float32)
    Ab = kim.Tensor(nd.array(_Ab), device=device)
    Bb = kim.Tensor(nd.array(_Bb), device=device)

    np.testing.assert_allclose(_Ab @ _B, (Ab @ B).numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(_A @ _Bb, (A @ Bb).numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(_Ab @ _Bb, (Ab @ Bb).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=CPU_CUDA)
def test_power(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    _B = np.random.randint(1)
    A = kim.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(_A**_B, (A**_B).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=CPU_CUDA)
def test_log(shape, device):
    _A = np.random.randn(*shape).astype(np.float32) + 5.
    A = kim.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(np.log(_A), kim.log(A).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=CPU_CUDA)
def test_exp(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = kim.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(np.exp(_A), kim.exp(A).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=CPU_CUDA)
def test_relu(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = kim.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(np.maximum(_A, 0), kim.relu(A).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=CPU_CUDA)
def test_tanh(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = kim.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(np.tanh(_A), kim.tanh(A).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", GENERAL_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=CPU_CUDA)
def test_tanh_backward(shape, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = kim.Tensor(nd.array(_A), device=device)
    backward_check(kim.tanh, A)


STACK_PARAMETERS = [((5, 5), 0, 1),
    ((5, 5), 0, 2),
    ((1,5,7), 2, 5), ((2, 3), 1, 2)]
@pytest.mark.parametrize("shape, axis, l", STACK_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=CPU_CUDA)
def test_stack_forward(shape, axis, l, device):
    _A = [np.random.randn(*shape).astype(np.float32) for i in range(l)]
    A = [kim.Tensor(nd.array(_A[i]), device=device) for i in range(l)]
    A_t = [torch.Tensor(_A[i]) for i in range(l)]
    out = kim.stack(A, axis=axis)
    out_t = torch.stack(A_t, dim=axis)
    np.testing.assert_allclose(out_t.numpy(), out.numpy(), atol=1e-5, rtol=1e-5)
    
    B = kim.split(out, axis=axis)
    for i, a in enumerate(A):
        np.testing.assert_allclose(B[i].numpy(), a.numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape, axis, l", STACK_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=CPU_CUDA)
def test_stack_backward(shape, axis, l, device):
    x = [np.random.randn(*shape).astype(np.float32) for i in range(l)]
    A = [kim.Tensor(x[i], device=device) for i in range(l)]
    A_t = [torch.Tensor(x[i]) for i in range(l)]
    for x in A_t: x.requires_grad = True
    kim.stack(A, axis=axis).sum().backward()
    torch.stack(A_t, dim=axis).sum().backward()
    for i in range(l): 
        np.testing.assert_allclose(A_t[i].grad.numpy(), A[i].grad.numpy(), atol=1e-5, rtol=1e-5)


SPLIT_PARAMETERS = [((6, 5, 18), 2, 6),
    ((5, 5), 0, 1), ((6, 5), 0, 2),
    ((1,5,10), 2, 5), ((2, 6), 1, 3),
]

@pytest.mark.parametrize("shape, axis, l", SPLIT_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=CPU_CUDA)
def test_split_forward(shape, axis, l, device):
    print(">>>", shape, axis, l)
    x = np.random.randn(*shape).astype(np.float32)
    A = kim.Tensor(x, device=device)
    A_t = torch.Tensor(x); A_t.requires_grad = True

    b = kim.split(A, axis, chunks=l)    
    b_t = torch.split(A_t, x.shape[axis] // l, dim=axis)

    for i in range(l):
        np.testing.assert_allclose(b[i].numpy(), b_t[i].detach().numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape, axis, l", SPLIT_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=CPU_CUDA)
def test_split_backward(shape, axis, l, device):
    x = np.random.randn(*shape).astype(np.float32)
    A = kim.Tensor(x, device=device)
    A_t = torch.Tensor(x); A_t.requires_grad = True

    b = kim.split(A, axis, chunks=l)    
    b_t = torch.split(A_t, x.shape[axis] // l, dim=axis)

    # for i in range(l):
    i = 0
    b[i].sum().backward()
    b_t[i].sum().backward()
    np.testing.assert_allclose(A_t.grad.numpy(), A.grad.numpy(), atol=1e-5, rtol=1e-5)


SUMMATION_PARAMETERS = [((1, 1, 1), None),
    ((5, 3), 0),
    ((8, 3, 2), 1),
    ((8, 3, 2), 2)
]
@pytest.mark.parametrize("shape, axes", SUMMATION_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=CPU_CUDA)
def test_summation(shape, axes, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = kim.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(np.sum(_A, axes), kim.summation(A, axes=axes).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape, axes", SUMMATION_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=CPU_CUDA)
def test_summation_backward(shape, axes, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = kim.Tensor(nd.array(_A), device=device)
    backward_check(kim.summation, A, axes=axes, keepdims=True)


BROADCAST_SHAPES = [((1, 1, 1), (3, 3, 3)),
    ((4, 1, 6), (4, 3, 6))]
@pytest.mark.parametrize("shape,shape_to", BROADCAST_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=CPU_CUDA)
def test_broadcast_to(shape, shape_to, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = kim.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(np.broadcast_to(_A, shape_to), kim.broadcast_to(A, shape_to).numpy(), atol=1e-5, rtol=1e-5)


RESHAPE_SHAPES = [((1, 1, 1), (1,)),
    ((4, 1, 6), (6, 4, 1))]
@pytest.mark.parametrize("shape,shape_to", RESHAPE_SHAPES)
@pytest.mark.parametrize("device", _DEVICES, ids=CPU_CUDA)
def test_reshape(shape, shape_to, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = kim.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(np.reshape(_A, shape_to), kim.reshape(A, shape_to).numpy(), atol=1e-5, rtol=1e-5)


TRANSPOSE_SHAPES = [(1, 1, 1), (4, 5, 6)]
TRANSPOSE_AXES = [(0, 1), (0, 2), None]
@pytest.mark.parametrize("shape", TRANSPOSE_SHAPES)
@pytest.mark.parametrize("axes", TRANSPOSE_AXES)
@pytest.mark.parametrize("device", _DEVICES, ids=CPU_CUDA)
def test_transpose(shape, axes, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = kim.Tensor(nd.array(_A), device=device)
    if axes is None:
        np_axes = (_A.ndim - 2, _A.ndim - 1)
    else:
        np_axes = axes
    np.testing.assert_allclose(np.swapaxes(_A, np_axes[0], np_axes[1]), kim.transpose(A, axes=axes).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape, axes", SUMMATION_PARAMETERS)
@pytest.mark.parametrize("device", _DEVICES, ids=CPU_CUDA)
def test_logsumexp(shape, axes, device):
    _A = np.random.randn(*shape).astype(np.float32)
    A = kim.Tensor(nd.array(_A), device=device)
    A_t = torch.Tensor(_A)
    if axes is None:
        t_axes = tuple(list(range(len(shape))))
    else:
        t_axes = axes
    np.testing.assert_allclose(torch.logsumexp(A_t, dim=t_axes).numpy(), kim.logsumexp(A, axes=axes).numpy(), atol=1e-5, rtol=1e-5)
