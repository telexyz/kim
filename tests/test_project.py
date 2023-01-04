import pytest
import numpy as np

import kim
import torch
import pytest


def get_tensor(*shape, entropy=1):
    np.random.seed(np.prod(shape) * len(shape) * entropy)
    return kim.Tensor(np.random.randint(0, 100, size=shape) / 20, dtype="float32")


def get_int_tensor(*shape, low=0, high=10, entropy=1):
    np.random.seed(np.prod(shape) * len(shape) * entropy)
    return kim.Tensor(np.random.randint(low, high, size=shape))


@pytest.mark.parametrize("rows", [5, 1, 200, 1000, 2827])
@pytest.mark.parametrize("classes", [10, 2, 30, 99])
def test_softmax_loss(rows, classes, eps=1e-05):
    x = get_tensor(rows, classes)
    y = get_int_tensor(rows, low=0, high=classes)
    f = kim.nn.SoftmaxLoss()
    loss = f(x, y) # to compare to torch

    f_ = torch.nn.CrossEntropyLoss()
    x_ = torch.Tensor(x.numpy())
    x_.requires_grad = True
    y_ = torch.Tensor(y.numpy()).long()
    loss_ = f_(x_, y_)

    np.testing.assert_allclose(loss.numpy(), loss_.detach().numpy(), rtol=eps, atol=eps)

    # Backward
    loss.backward()
    loss_.backward()
    np.testing.assert_allclose(x.grad.numpy(),  x_.grad.numpy(), rtol=eps, atol=eps)


@pytest.mark.parametrize("N", [1, 2])
@pytest.mark.parametrize("C", [1, 3])
@pytest.mark.parametrize("H", [2, 6, 16])
@pytest.mark.parametrize("W", [2, 4, 100])
def test_max_pooling(N,H,W,C):
    X = kim.init.randn(N, C, H, W, requires_grad=True)
    X_ = torch.Tensor(X.numpy())
    X_.requires_grad = True

    # Forward
    res_ = torch.nn.MaxPool2d((2, 1))(X_)
    X = X.transpose(axes=(1, 2)).transpose(axes=(2, 3)) # N,C,H,W => N,H,W,C
    res = kim.nn.MaxPool2x1()(X)  # N,H,W,C => N,C,H,W
    np.testing.assert_allclose(res_.detach().numpy(), res.transpose(
        axes=(2, 3)).transpose(axes=(1, 2)).numpy(), rtol=1e-04, atol=1e-04)

    # backward
    res_.sum().backward()
    res.sum().backward()
    np.testing.assert_allclose(X_.grad.numpy(), X.grad.transpose(
        axes=(2, 3)).transpose(axes=(1, 2)).numpy(), rtol=1e-04, atol=1e-04)


@pytest.mark.parametrize("N", [2])
@pytest.mark.parametrize("H", [4, 6, 16])
@pytest.mark.parametrize("W", [1, 3, 100])
@pytest.mark.parametrize("C_in", [5])
@pytest.mark.parametrize("C_out", [1])
@pytest.mark.parametrize("kh", [5, 1, 3])
@pytest.mark.parametrize("kw", [1, 4, 9])
@pytest.mark.parametrize("sw", [1, 2, 5])
@pytest.mark.parametrize("sh", [1, 3])
@pytest.mark.parametrize("dw", [1])
@pytest.mark.parametrize("dh", [1])
def test_conv_kernel_hw(N, H, W, C_in, C_out, kh, kw, sw, sh, dw, dh):
    X = kim.init.randn(N, C_in, H, W, requires_grad=True)
    X_ = torch.Tensor(X.numpy())
    X_.requires_grad = True

    imp_kim = kim.nn.Conv(C_in, C_out, (kh, kw), stride=(sh, sw), dilation=(dh, dw))
    imp_torch = torch.nn.Conv2d(C_in, C_out, (kh, kw), stride=(sh, sw), dilation=(dh, dw), padding=(dh*kh//2, dw*kw//2))

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
