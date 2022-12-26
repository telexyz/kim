import pytest
import numpy as np

import kim
import torch
import pytest

@pytest.mark.parametrize("N", [1, 2])
@pytest.mark.parametrize("H", [2, 6, 16])
@pytest.mark.parametrize("W", [2, 4, 100])
@pytest.mark.parametrize("C", [1, 3])
def test_max_pooling(N,H,W,C):
    X = kim.init.randn(N, C, H, W, requires_grad=True)
    X_ = torch.Tensor(X.numpy())
    X_.requires_grad = True

    res_ = torch.nn.MaxPool2d((2, 1))(X_)
    res = kim.nn.MaxPooling2x1()(X)

    # res_ = torch.nn.MaxPool2d((1, 2))(X_)
    # res = kim.ops.max_pooling_1x2(X)

    np.testing.assert_allclose(res_.detach().numpy(), res.numpy(), rtol=1e-04, atol=1e-04)

    # backward
    res_.sum().backward()
    res.sum().backward()
    np.testing.assert_allclose(X_.grad.numpy(), X.grad.numpy(), rtol=1e-04, atol=1e-04)


@pytest.mark.parametrize("N", [1, 2])
@pytest.mark.parametrize("H", [4, 6, 16])
@pytest.mark.parametrize("W", [1, 3, 100])
@pytest.mark.parametrize("C_in", [1, 5])
@pytest.mark.parametrize("C_out", [1, 32])
@pytest.mark.parametrize("kh", [5, 1, 3])
@pytest.mark.parametrize("kw", [1, 4, 9])
def test_conv_kernel_hw(N, H, W, C_in, C_out, kh, kw):
    X = kim.init.randn(N, C_in, H, W, requires_grad=True)
    X_ = torch.Tensor(X.numpy())
    X_.requires_grad = True

    imp_kim = kim.nn.Conv(C_in, C_out, (kh, kw))
    imp_torch = torch.nn.Conv2d(C_in, C_out, (kh, kw), padding=(kh//2, kw//2))

    # Ensure ndl and torch have same init params (transpose(3, 2, 0, 1) = KKIO -> OIKK)
    imp_torch.weight.data = torch.tensor(imp_kim.weight.numpy().transpose(3, 2, 0, 1))
    imp_torch.bias.data = torch.tensor(imp_kim.bias.numpy())

    # forward
    res_ = imp_torch(X_)
    res = imp_kim(X)

    np.testing.assert_allclose(res_.shape, res.shape)
    np.testing.assert_allclose(res_.detach().numpy(), res.numpy(), rtol=1e-04, atol=1e-04)

    # backward
    res_.sum().backward()
    res.sum().backward()
    np.testing.assert_allclose(X_.grad.numpy(),  X.grad.numpy(), rtol=1e-04, atol=1e-04)
