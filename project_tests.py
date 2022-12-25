import pytest
import numpy as np
# import sys
# sys.path.insert(0, './python')
# sys.path.insert(0, '../python')

import kim as ndl
import torch
import pytest

@pytest.mark.parametrize("N", [1,2])
@pytest.mark.parametrize("H", [1, 3, 6, 16])
@pytest.mark.parametrize("W", [2, 4, 100])
@pytest.mark.parametrize("C", [1,3])
def test_max_pooling(N,H,W,C):
# def test_max_pooling():
    N,C,H,W = 2, 1, 3, 4
    seed = 1
    X = np.random.default_rng(seed=seed).normal(size=(N, C, H, W))
    X_ = torch.Tensor(X)
    X_.requires_grad = True
    X = ndl.Tensor(X)
    imp_torch = torch.nn.MaxPool2d((1,2))
    res_torch = imp_torch(X_)
    res_ndl = ndl.ops.max_pooling_1x2(X)
 
    np.testing.assert_allclose(res_torch.detach().numpy(
    ), res_ndl.numpy(), rtol=1e-04, atol=1e-04)

    # backward
    res_torch.sum().backward()
    res_ndl.sum().backward()
    res_torch_grad = X_.grad.numpy()
    res_ndl_grad = X.grad.numpy()

    np.testing.assert_allclose(res_torch_grad, res_ndl_grad, rtol=1e-04, atol=1e-04)

    np.testing.assert_allclose(
        X_.grad.numpy(), X.grad.numpy(), rtol=1e-04, atol=1e-04)


@pytest.mark.parametrize("N", [1, 2])
@pytest.mark.parametrize("H", [4, 6, 16])
@pytest.mark.parametrize("W", [1, 3, 100])
@pytest.mark.parametrize("C_in", [1, 3, 5])
@pytest.mark.parametrize("C_out", [1, 16, 32])
@pytest.mark.parametrize("kh", [5, 1, 3])
@pytest.mark.parametrize("kw", [1, 4, 9])
def test_conv_kernel_hw(N, H, W, C_in, C_out, kh, kw):
    seed = 1
    X = np.random.default_rng(seed=seed).normal(size=(N, C_in, H, W))

    X_ = torch.Tensor(X)
    X_.requires_grad = True
    X = ndl.Tensor(X)
    # transpose X as well because in NN.Conv, use pytorch convention.

    imp_ndl = ndl.nn.Conv(C_in, C_out, (kh, kw))
    imp_torch = torch.nn.Conv2d(C_in, C_out, (kh, kw), padding=(kh//2, kw//2))
    imp_torch.weight.data = torch.tensor(imp_ndl.weight.cached_data.numpy().transpose(3, 2, 0, 1))
    imp_torch.bias.data = torch.tensor(imp_ndl.bias.cached_data.numpy())

    # forward
    res_torch = imp_torch(X_)
    res_ndl = imp_ndl(X)

    np.testing.assert_allclose(res_torch.shape, res_ndl.shape)
    np.testing.assert_allclose(res_torch.detach().numpy(), res_ndl.detach().numpy(), rtol=1e-04, atol=1e-04)

    # backward
    res_torch.sum().backward()
    res_ndl.sum().backward()
    # res_torch_grad = X_.grad.transpose(1, 2).transpose(2, 3).numpy()
    res_torch_grad = X_.grad.numpy()
    res_ndl_grad = X.grad.numpy()
    np.testing.assert_allclose(res_torch_grad, res_ndl_grad, rtol=1e-04, atol=1e-04)

    err1 = np.linalg.norm(X_.grad.numpy() - X.grad.numpy())
    assert err1 < 1e-3, "input grads match"
    # err2 = np.linalg.norm(Wtch.grad.numpy() - W.grad.numpy())
    # assert err2 < 1e-2, "weight grads match"
