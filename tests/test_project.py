import pytest
import numpy as np

import kim
import torch
import pytest

import sys; sys.path.append('project')
from project.myexp import mymodel

def get_tensor(*shape, entropy=1):
    np.random.seed(np.prod(shape) * len(shape) * entropy)
    return kim.Tensor(np.random.randint(0, 100, size=shape) / 20, dtype="float32")


def get_int_tensor(*shape, low=0, high=10, entropy=1):
    np.random.seed(np.prod(shape) * len(shape) * entropy)
    return kim.Tensor(np.random.randint(low, high, size=shape))

'''
(0): Conv2d(1, 64, kernel_size=(5, 3), stride=(1, 1), padding=(2, 1))
(1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(2): LeakyReLU(negative_slope=0.01)
(3): MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0, dilation=1, ceil_mode=False)
(4): Conv2d(64, 128, kernel_size=(5, 3), stride=(1, 1), padding=(2, 1))
(5): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(6): LeakyReLU(negative_slope=0.01)
(7): MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0, dilation=1, ceil_mode=False)
(8): Flatten(start_dim=1, end_dim=-1)
(9): Linear(in_features=15360, out_features=2, bias=True)
'''


@pytest.mark.parametrize("batch_size", [5, 1, 256, 1024, 2048])
def test_model(batch_size, eps=1e-04):
    # from myexp import mymodel, torch, kim
    model = mymodel(kim.nn, dropout=False)
    model_ = mymodel(torch.nn, dropout=False)
    
    ## Assign same weights between models
    for i, x in enumerate(model):
        if isinstance(x, kim.nn.Conv): # i=0; model[i]
            model_[i].weight.data = torch.tensor(x.weight.numpy().transpose(3, 2, 0, 1))
            model_[i].bias.data = torch.tensor(x.bias.numpy())
        if isinstance(x, kim.nn.BatchNorm2d): # i=1; model[i]
            model_[i].weight.data = torch.tensor(x.weight.numpy().reshape((kim.prod(x.weight.shape),)))
            model_[i].bias.data = torch.tensor(x.bias.numpy().reshape((kim.prod(x.bias.shape),)))
        if isinstance(x, kim.nn.Linear): # i=9; model[i]
            model_[i].weight.data = torch.tensor(x.weight.numpy().transpose())
            model_[i].bias.data = torch.tensor(x.bias.numpy())

    # (B, 1, 32, 15), 1-channel, 32x15 image
    x = kim.init.randn(batch_size, 1, 32, 15, requires_grad=True)
    x_ = torch.Tensor(x.numpy())

    y = model(x)
    y_ = model_(x_)
    np.testing.assert_allclose(y.numpy(), y_.detach().numpy(), rtol=eps, atol=eps)

    B, classes = y.shape
    target = get_int_tensor(B, low=0, high=classes)
    target_ = torch.Tensor(target.numpy()).long()

    loss = kim.nn.SoftmaxLoss()(y, target)
    loss_ = torch.nn.CrossEntropyLoss()(y_, target_)

    diff = abs(loss.numpy().sum() - loss_.detach().numpy().sum())
    print(">>> loss diff", diff)
    assert diff < eps

    # torch.optim.Adam(model_.parameters(), lr=1e-5).zero_grad()
    # for x in model_.parameters(): x.requires_grad = True
    model_[0].weight.requires_grad = True
    loss.backward()
    loss_.backward()

    z, z_ = model[0].weight, model_[0].weight
    np.testing.assert_allclose(z.grad.numpy().transpose((3, 2, 0, 1)), z_.grad.numpy(), rtol=eps, atol=eps)
    z, z_ = model[0].bias, model_[0].bias
    np.testing.assert_allclose(z.grad.numpy(), z_.grad.numpy(), rtol=eps, atol=eps)

    z, z_ = model[4].weight, model_[4].weight
    np.testing.assert_allclose(z.grad.numpy().transpose((3, 2, 0, 1)), z_.grad.numpy(), rtol=eps*10, atol=eps*10)
    z, z_ = model[4].bias, model_[4].bias
    np.testing.assert_allclose(z.grad.numpy(), z_.grad.numpy(), rtol=eps, atol=eps)

    z, z_ = model[9].weight, model_[9].weight
    np.testing.assert_allclose(z.grad.numpy().transpose(), z_.grad.numpy(), rtol=eps, atol=eps)
    z, z_ = model[9].bias, model_[9].bias
    np.testing.assert_allclose(z.grad.numpy(), z_.grad.numpy(), rtol=eps, atol=eps)



@pytest.mark.parametrize("rows", [5, 1, 200, 1000, 2827])
@pytest.mark.parametrize("classes", [10, 2, 30, 99])
def test_softmax_loss(rows, classes, eps=1e-05):
    # x = kim.init.randn(rows, classes, requires_grad=True)
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
