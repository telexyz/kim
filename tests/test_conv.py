import sys
sys.path.append('./python')
import numpy as np
import pytest
from kim import backend_ndarray as nd
import kim as kim
import mugrade
import itertools


_DEVICES = [kim.cpu(), pytest.param(kim.cuda(),
    marks=pytest.mark.skipif(not kim.cuda().enabled(), reason="No GPU"))]

def backward_check(f, *args, **kwargs):
    eps = 1e-3
    out = f(*args, **kwargs)
    c = np.random.randn(*out.shape)
    
    is_stacked = isinstance(args[0], list)
    if is_stacked: args = args[0]
    print(">>>", is_stacked, len(args), args[0].shape)

    numerical_grad = [np.zeros(a.shape) for a in args]
    num_args = len(args)

    for i in range(num_args):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            if is_stacked:
                f1 = (f(args, **kwargs).numpy() * c).sum()
            else:
                f1 = (f(*args, **kwargs).numpy() * c).sum()

            args[i].realize_cached_data().flat[j] -= 2 * eps
            if is_stacked:
                f2 = (f(args, **kwargs).numpy() * c).sum()
            else:
                f2 = (f(*args, **kwargs).numpy() * c).sum()

            args[i].realize_cached_data().flat[j] += eps
            numerical_grad[i].flat[j] = (f1 - f2) / (2 * eps)

    backward_grad = out.op.gradient(kim.Tensor(c, device=args[0].device), out)

    if isinstance(backward_grad[0], kim.TensorTuple): # TODO keep this?
        backward_grad = backward_grad[0].tuple()

    error = sum(
        np.linalg.norm(backward_grad[i].numpy() - numerical_grad[i])
        for i in range(len(args))
    )
    assert error < 1e-2
    return [g.numpy() for g in backward_grad]


stack_back_params = [
    ( (3, 4), 3, 0),
    ( (3, 4), 3, 1),
    ( (3, 4), 3, 2),
    ( (3, 4), 1, 2),
    ( (3, 4), 5, 2),
    ( (3,), 1, 0),
]
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("shape, n, axis", stack_back_params)
def test_stack_backward(shape, n, axis, device):
    np.random.seed(0)
    get_tensor = lambda shape: kim.Tensor(np.random.randn(*shape)*5, device=device)
    backward_check(kim.stack, [get_tensor(shape) for _ in range(n)], axis=axis)


stack_params = [
    {"shape": (10,3),    "n": 4, "axis": 0},
    {"shape": (4, 5, 6), "n": 5, "axis": 0},
    {"shape": (4, 5, 6), "n": 3, "axis": 1},
    {"shape": (4, 5, 6), "n": 2, "axis": 2},
    {"shape": (4, 5, 6), "n": 2, "axis": 3},
]
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("params", stack_params)
def test_stack_forward(params, device):
    np.random.seed(0)
    shape, n, axis = params['shape'], params['n'], params['axis']
    to_stack_kim = []
    to_stack_npy = []
    for i in range(n):
        _A = np.random.randn(*shape)
        to_stack_kim += [kim.Tensor(_A, device=device)]
        to_stack_npy += [_A]

    lhs = np.stack(to_stack_npy, axis=axis)
    rhs = kim.stack(to_stack_kim, axis=axis)


pad_params = [
    {"shape": (10, 32, 32, 8), "padding": ( (0, 0), (2, 2), (2, 2), (0, 0) )},
    {"shape": (10, 32, 32, 8), "padding": ( (0, 0), (0, 0), (0, 0), (0, 0) )},
    {"shape": (10, 32, 32, 8), "padding": ( (0, 2), (1, 10), (3, 8), (2, 12) )},
]
@pytest.mark.parametrize("device", [nd.cpu()])
@pytest.mark.parametrize("params", pad_params)
def test_pad_forward(params, device):
    np.random.seed(0)
    shape, padding = params['shape'], params['padding']
    _A = np.random.randn(*shape)
    _B = np.pad(_A, padding)
    A = nd.NDArray(_A, device=device)
    B = A.pad(padding)

    assert np.linalg.norm(A.numpy() - _A) < 1e-4


flip_forward_params = [
    {"shape": (10, 5), "axes": (0,)},
    {"shape": (10, 5), "axes": (1,)},
    {"shape": (10, 5), "axes": (0,1)},
    {"shape": (10, 32, 32, 8), "axes": (0,1)},
    {"shape": (3, 3, 6, 8), "axes": (0,1)},
    {"shape": (10, 32, 32, 8), "axes": (1,2)},
    {"shape": (3, 3, 6, 8), "axes": (1,2)},
    {"shape": (10, 32, 32, 8), "axes": (2,3)},
    {"shape": (3, 3, 6, 8), "axes": (2,3)},
    {"shape": (10, 32, 32, 8), "axes": (0,1,2,3)},
]
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("params", flip_forward_params)
def test_flip_forward(params, device):
    np.random.seed(0)
    shape, axes = params['shape'], params['axes']
    _A = np.random.randn(*shape)
    _B = np.flip(_A, axes)
    A = kim.Tensor(_A, device=device)
    B = kim.flip(A, axes=axes)

    assert np.linalg.norm(A.numpy() - _A) < 1e-4


flip_backward_params = [
    {"shape": (10, 5), "axes": (0,)},
    {"shape": (10, 5), "axes": (1,)},
    {"shape": (10, 5), "axes": (0,1)},
    {"shape": (2, 3, 3, 8), "axes": (0,1)},
    {"shape": (3, 3, 6, 4), "axes": (0,1)},
    {"shape": (2, 3, 3, 4), "axes": (1,2)},
    {"shape": (3, 3, 6, 4), "axes": (1,2)},
    {"shape": (2, 3, 3, 4), "axes": (2,3)},
    {"shape": (3, 3, 6, 4), "axes": (2,3)},
    {"shape": (2, 3, 3, 4), "axes": (0,1,2,3)},
]
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("params", flip_backward_params)
def test_flip_backward(params, device):
    np.random.seed(0)
    shape, axes = params['shape'], params['axes']
    backward_check(kim.flip, kim.Tensor(np.random.randn(*shape), device=device), axes=axes)


# @pytest.mark.parametrize("device", _DEVICES)
# def test_init_calculate_fans(device):
#     _A = np.random.randn(3, 3, 16, 8)
#     A = kim.Tensor(_A, device=device)
#     assert kim.init._calculate_fans(A) == (144, 72)

#     _A = np.random.randn(3, 3, 16, 8)
#     A = kim.Tensor(_A, device=device)
#     assert kim.init._calculate_fans(A) == (144, 72)


#     _A = np.random.randn(16, 8)
#     A = kim.Tensor(_A, device=device)
#     assert kim.init._calculate_fans(A) == (16, 8)


@pytest.mark.parametrize("device", _DEVICES)
def test_init_kaiming_uniform(device):
    _A = np.random.randn(3, 3, 16, 8)
    A = kim.Tensor(_A, device=device)
    np.random.seed(0)
    A = kim.init.kaiming_uniform(16*9, 8*9, shape=A.shape)
    assert abs(A.sum().numpy() - -2.5719218) < 1e-4


@pytest.mark.parametrize("device", _DEVICES)
def test_resnet9(device):
    def num_params(model):
        return np.sum([np.prod(x.shape) for x in model.parameters()])

    from apps.models import ResNet9
    np.random.seed(0)
    model = ResNet9(device=device)

    assert num_params(model) == 431946

    _A = np.random.randn(2, 3, 32, 32)
    A = kim.Tensor(_A, device=device)
    y = model(A)
    print(">>> test_resnet9:", A.shape, "->", y.shape)
    assert np.linalg.norm(np.array([
        [-1.8912625 ,  0.64833605,  1.9400386 ,  1.1435282 ,  1.89777   ,
         2.9039745 , -0.10433993,  0.35458302, -0.5684191 ,  2.6178317 ],
        [-0.2905612 , -0.4147861 ,  0.90268034,  0.46530387,  1.3335679 ,
         1.8534894 , -0.1867125 , -2.4298222 , -0.5344223 ,  4.362149  ]]) - y.numpy()) < 1e-2



@pytest.mark.parametrize("device", _DEVICES)
def test_dilate_forward(device):
    np.random.seed(0)
    device = kim.cpu()

    _A = np.random.randint(1, 10, size=(2, 5))
    A = kim.Tensor(_A, device=device)
    assert np.linalg.norm(kim.dilate(A, dilation=0, axes=(0,)).numpy() - np.array(
      [[6., 1., 4., 4., 8.],
       [4., 6., 3., 5., 8.]])) < 1e-5 

    _A = np.random.randint(1, 10, size=(2, 5))
    A = kim.Tensor(_A, device=device)
    assert np.linalg.norm(kim.dilate(A, dilation=1, axes=(0,)).numpy() - np.array(
      [[7., 9., 9., 2., 7.],
       [0., 0., 0., 0., 0.],
       [8., 8., 9., 2., 6.],
       [0., 0., 0., 0., 0.]])) < 1e-5

    _A = np.random.randint(1, 10, size=(2, 5))
    A = kim.Tensor(_A, device=device)
    assert np.linalg.norm(kim.dilate(A, dilation=1, axes=(1,)).numpy() - np.array(
      [[9., 0., 5., 0., 4., 0., 1., 0., 4., 0.],
       [6., 0., 1., 0., 3., 0., 4., 0., 9., 0.]])) < 1e-5

    _A = np.random.randint(1, 10, size=(2, 5))
    A = kim.Tensor(_A, device=device)
    assert np.linalg.norm(kim.dilate(A, dilation=1, axes=(0,1)).numpy() - np.array(
      [[2., 0., 4., 0., 4., 0., 4., 0., 8., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 0., 2., 0., 1., 0., 5., 0., 8., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])) < 1e-5

    _A = np.random.randint(1, 10, size=(2, 2))
    A = kim.Tensor(_A, device=device)
    assert np.linalg.norm(kim.dilate(A, dilation=2, axes=(0,1)).numpy() - np.array(
      [[4., 0., 0., 3., 0., 0.],
       [0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0.],
       [8., 0., 0., 3., 0., 0.],
       [0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0.]])) < 1e-5

    _A = np.random.randint(1, 10, size=(2, 2, 2, 2))
    A = kim.Tensor(_A, device=device)
    assert np.linalg.norm(kim.dilate(A, dilation=1, axes=(1,2)).numpy() - np.array([[[[1., 1.],
         [0., 0.],
         [5., 6.],
         [0., 0.]],

        [[0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.]],

        [[6., 7.],
         [0., 0.],
         [9., 5.],
         [0., 0.]],

        [[0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.]]],


       [[[2., 5.],
         [0., 0.],
         [9., 2.],
         [0., 0.]],

        [[0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.]],

        [[2., 8.],
         [0., 0.],
         [4., 7.],
         [0., 0.]],

        [[0., 0.],
         [0., 0.],
         [0., 0.],
         [0., 0.]]]])) < 1e-5


dilate_backward_params = [
    {"shape": (2, 5),          "d": 1, "axes": (0,)},
    {"shape": (2, 5),          "d": 2, "axes": (1,)},
    {"shape": (2, 5),          "d": 1, "axes": (0,1)},
    {"shape": (2, 5),          "d": 0, "axes": (0,1)},
    {"shape": (2, 3, 3, 4),     "d": 2, "axes": (0,1)},
    {"shape": (3, 3, 6, 4),     "d": 3, "axes": (0,1)},
    {"shape": (2, 3, 3, 4),     "d": 0, "axes": (1,2)},
    {"shape": (2, 3, 3, 4),     "d": 1, "axes": (1,2)},
    {"shape": (3, 3, 6, 4),     "d": 1, "axes": (1,2)},
    {"shape": (2, 3, 3, 4),     "d": 1, "axes": (2,3)},
    {"shape": (3, 3, 6, 4),     "d": 1, "axes": (2,3)},
    {"shape": (2, 3, 3, 4),     "d": 1, "axes": (0,1,2,3)},
]
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("params", dilate_backward_params)
def test_dilate_backward(params, device):
    np.random.seed(0)
    shape, d, axes = params['shape'], params['d'], params['axes']
    backward_check(kim.dilate, kim.Tensor(np.random.randn(*shape), device=device), dilation=d, axes=axes)


def test_stack_vs_pytorch():
    np.random.seed(0)
    import torch
    A = np.random.randn(5, 5)
    B = np.random.randn(5, 5)
    C = np.random.randn(5, 5)
    D = np.random.randn(15, 5)

    Akim = kim.Tensor(A, requires_grad=True)
    Bkim = kim.Tensor(B, requires_grad=True)
    Ckim = kim.Tensor(C, requires_grad=True)
    Dkim = kim.Tensor(D, requires_grad=True)

    Atch = torch.tensor(A, requires_grad=True)
    Btch = torch.tensor(B, requires_grad=True)
    Ctch = torch.tensor(C, requires_grad=True)
    Dtch = torch.tensor(D, requires_grad=True)

    Xkim = kim.stack([Akim, Ckim @ Bkim, Ckim], axis=1)
    Xtch = torch.stack([Atch, Ctch @ Btch, Ctch], dim=1)

    assert Xkim.shape == Xtch.shape
    assert np.linalg.norm(Xkim.numpy() - Xtch.detach().numpy()) < 1e-3

    Ykim = (Dkim @ Xkim.reshape((5, 15)) @ Dkim).sum()
    Ytch = (Dtch @ Xtch.reshape(5, 15) @ Dtch).sum()

    assert np.linalg.norm(Ykim.numpy() - Ytch.detach().numpy()) < 1e-3

    Ykim.backward()
    Ytch.backward()

    assert np.linalg.norm(Akim.grad.cached_data.numpy() - Atch.grad.detach().numpy()) < 1e-3
    assert np.linalg.norm(Bkim.grad.cached_data.numpy() - Btch.grad.detach().numpy()) < 1e-3
    assert np.linalg.norm(Ckim.grad.cached_data.numpy() - Ctch.grad.detach().numpy()) < 1e-3



conv_forward_params = [
    (4, 8, 16, 3, 1),
    (32, 8, 16, 3, 2),
    (32, 8, 8, 3, 2),
    (32, 16, 8, 3, 1),
    (32, 16, 8, 3, 2)
]
@pytest.mark.parametrize("s,cin,cout,k,stride", conv_forward_params)
@pytest.mark.parametrize("device", _DEVICES)
def test_nn_conv_forward(s, cin, cout, k, stride, device):
    np.random.seed(0)
    import torch
    f = kim.nn.Conv(cin, cout, k, stride=stride, device=device)
    x = kim.init.rand(10, cin, s, s, device=device)

    g = torch.nn.Conv2d(cin, cout, k, stride=stride, padding=k//2)
    g.weight.data = torch.tensor(f.weight.cached_data.numpy().transpose(3, 2, 0, 1))
    g.bias.data = torch.tensor(f.bias.cached_data.numpy())
    z = torch.tensor(x.cached_data.numpy())

    assert np.linalg.norm(f(x).cached_data.numpy() - g(z).data.numpy()) < 1e-3


conv_back_params = [
    (4, 1, 1, 3, 1),
    (14, 8, 16, 3, 1),
    (14, 8, 16, 3, 2),
    (14, 8, 8, 3, 1),
    (14, 8, 8, 3, 2),
    (14, 16, 8, 3, 1),
    (14, 16, 8, 3, 2),
]
@pytest.mark.parametrize("s,cin,cout,k,stride", conv_back_params)
@pytest.mark.parametrize("device", _DEVICES)
def test_nn_conv_backward(s, cin, cout, k, stride, device):
    np.random.seed(0)
    import torch
    f = kim.nn.Conv(cin, cout, k, stride=stride, device=device)
    # 
    g = torch.nn.Conv2d(cin, cout, k, stride=stride, padding=k//2)
    g.weight.data = torch.tensor(f.weight.cached_data.numpy().transpose(3, 2, 0, 1))
    g.bias.data = torch.tensor(f.bias.cached_data.numpy())

    x = kim.init.rand(1, cin, s, s, device=device, requires_grad=True)
    z = torch.tensor(x.cached_data.numpy(), requires_grad=True)
    z.requires_grad = True

    y1 = f(x).sum()
    y2 = g(z).sum()
    # 
    y1.backward()
    y2.backward()

    # assert np.linalg.norm(g.weight.grad.data.numpy() - f.weight.grad.cached_data.numpy().transpose(3, 2, 0, 1)) < 1e-3, "weight gradients match"
    assert np.linalg.norm(g.bias.grad.data.numpy() - f.bias.grad.cached_data.numpy()) < 1e-3, "bias gradients match"
    assert np.linalg.norm(z.grad.data.numpy() - x.grad.cached_data.numpy()) < 1e-3, "input gradients match"


op_conv_shapes = [
    ( (3, 14, 14, 8), (3, 3, 8, 16), 1, 0 ),
    ( (3, 14, 14, 8), (3, 3, 8, 16), 1, 1 ),
    ( (3, 16, 16, 8), (3, 3, 8, 16), 1, 2 ),
    ( (3, 16, 16, 8), (3, 3, 8, 14), 1, 0 ),
    ( (3, 16, 16, 2), (3, 3, 2, 14), 1, 0 ),

    ( (3, 14, 14, 8), (3, 3, 8, 16), 2, 0 ),
    ( (3, 14, 14, 8), (3, 3, 8, 16), 2, 1 ),
    ( (3, 16, 16, 8), (3, 3, 8, 16), 2, 2 ),
    ( (3, 16, 16, 8), (3, 3, 8, 14), 2, 0 ),
    ( (3, 16, 16, 2), (3, 3, 2, 14), 2, 0 ),

    ( (3, 16, 16, 24), (3, 3, 24, 14), 1, 0 ),
    ( (3, 14, 14, 8), (5, 5, 8, 16),   1, 0 ),
    ( (3, 17, 17, 8), (5, 5, 8, 16),   1, 0 ),
    ( (3, 17, 17, 1), (5, 5, 1, 16) ,  1, 0),
    ( (3, 17, 17, 16), (5, 5, 16, 1),  1, 0 ),
    ( (3, 17, 17, 16), (1, 1, 16, 1),  1, 0 ),
    ( (1, 14, 14, 2), (3, 3, 2, 2),    1, 0 ),
]
@pytest.mark.parametrize("Z_shape, W_shape, stride, padding", op_conv_shapes)
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("backward", [True, False], ids=["backward", "forward"])
def test_op_conv(Z_shape, W_shape, stride, padding, backward, device):
    np.random.seed(0)
    import torch
    _Z = np.random.randn(*Z_shape)*5
    _Z = _Z.astype(np.float32)
    _W = np.random.randn(*W_shape)*5
    _W = _W.astype(np.float32)
    Z = kim.Tensor(_Z, device=device)
    W = kim.Tensor(_W, device=device)
    y = kim.conv(Z, W, padding=padding, stride=stride)
    y2 = y.sum()
    if backward:
        y2.backward()
    Ztch = torch.Tensor(_Z).float()
    Ztch.requires_grad=True
    Wtch = torch.Tensor(_W).float()
    Wtch.requires_grad=True
    out = torch.nn.functional.conv2d(Ztch.permute(0, 3, 1, 2), Wtch.permute(3, 2, 0, 1), padding=padding, stride=stride)
    out2 = out.sum()
    if backward:
        out2.backward()
    if backward:
        err1 = np.linalg.norm(Ztch.grad.numpy() - Z.grad.numpy())
        err2 = np.linalg.norm(Wtch.grad.numpy() - W.grad.numpy())
    err3 = np.linalg.norm(out2.detach().numpy() - y2.numpy())
    if backward:
        assert err1 < 1e-2, "input grads match"
        assert err2 < 1e-2, "weight grads match"
    assert err3 < 1e-1, "outputs match %s, %s" % (y2, out2)


@pytest.mark.parametrize("device", _DEVICES)
def test_train_cifar10(device):
    np.random.seed(0)
    dataset = kim.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
    dataloader = kim.data.DataLoader(\
             dataset=dataset,
             batch_size=128,
             shuffle=False,
             # collate_fn=kim.data.collate_ndarray,
             # drop_last=False,
             # device=device,
             # dtype="float32"
             )
    from apps.models import ResNet9
    np.random.seed(0)
    model = ResNet9(device=device, dtype="float32")
    out = one_iter_of_cifar10_training(dataloader, model, niter=1, opt=kim.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001), device=device)
    assert np.linalg.norm(np.array(list(out)) - np.array([0.09375, 3.5892258])) < 1e-2


def one_iter_of_cifar10_training(dataloader, model, niter=1, loss_fn=kim.nn.SoftmaxLoss(), opt=None, device=None):
    np.random.seed(4)
    model.train()
    correct, total_loss = 0, 0
    i = 1
    for batch in dataloader:
        opt.reset_grad()
        X, y = batch
        X, y = kim.Tensor(X, device=device), kim.Tensor(y, device=device)
        out = model(X)
        correct += np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())
        loss = loss_fn(out, y)
        total_loss += loss.data.numpy() * y.shape[0]
        loss.backward()
        opt.step()
        if i >= niter:
            break
        i += 1
    return correct/(y.shape[0]*niter), total_loss/(y.shape[0]*niter)


######################    |    ######################
###################### MUGRADE ######################
######################    v    ######################

def Prepare(A):
    return (A.numpy().flatten()[:64], A.shape)


def Rand(*shape, device=kim.cpu(), entropy=1):
    np.random.seed(np.prod(shape) * len(shape) * entropy)
    _A = np.random.randint(low=1, high=10, size=shape)
    return kim.Tensor(_A, device=device)


def RandC(*shape, entropy=1):
    if kim.cuda().enabled():
        return Rand(*shape, device=kim.cuda(), entropy=2)
    else:
        raise NotImplementedError("You need a GPU to run these tests.")


def MugradeSubmit(things):
    mugrade.submit(Prepare(things))
    #print(Prepare(things))


def submit_conv_forward():
    def DoConvOp(batches, cin, cout, n, k=3, stride=1, padding=0, device=kim.cpu()):
        X = Rand(batches, n, n, cin, device=device)
        W = Rand(k, k, cin, cout, device=device)
        y = kim.conv(X, W, stride=stride, padding=padding)
        return y

    def DoConvLayer(batches, cin, cout, n, k=3, stride=1, bias=True, device=kim.cpu()):
        X = Rand(batches, cin, n, n, device=device)
        f = kim.nn.Conv(cin, cout, k, stride=stride, bias=bias, device=device)
        return f(X)

    MugradeSubmit(DoConvOp(2, 1, 2, 4, k=1, stride=1, padding=0))
    MugradeSubmit(DoConvOp(2, 1, 2, 4, k=1, stride=1, padding=2))
    MugradeSubmit(DoConvOp(2, 3, 1, 6, k=1, stride=2, padding=2))


    MugradeSubmit(DoConvOp(2, 1, 2, 4, k=3, stride=1, padding=0))
    MugradeSubmit(DoConvOp(3, 1, 2, 4, k=3, stride=1, padding=2))
    MugradeSubmit(DoConvOp(1, 1, 3, 6, k=5, stride=2, padding=2))

    MugradeSubmit(DoConvLayer(3, 2, 4, 6, k=3, stride=1, bias=True))
    MugradeSubmit(DoConvLayer(3, 4, 2, 6, k=3, stride=1, bias=True))
    MugradeSubmit(DoConvLayer(1, 1, 1, 12, k=3, stride=2, bias=True))
    MugradeSubmit(DoConvLayer(1, 1, 1, 12, k=1, stride=1, bias=False))
    MugradeSubmit(DoConvLayer(1, 2, 1, 12, k=7, stride=1, bias=False))
    MugradeSubmit(DoConvLayer(1, 1, 3, 12, k=7, stride=4, bias=False))


    if kim.cuda().enabled():
        MugradeSubmit(DoConvLayer(3, 2, 4, 6, k=3, stride=1, bias=False, device=kim.cuda()))
        MugradeSubmit(DoConvLayer(3, 4, 2, 6, k=3, stride=1, bias=False, device=kim.cuda()))
    else:
        print('You need a GPU to run these tests!')


def submit_conv_backward():

    def DoConvOpBackward(batches, cin, cout, n, k=3, stride=1, padding=0, device=kim.cpu(), wrtX=True):
        X = Rand(batches, n, n, cin, device=device)
        X.requires_grad = True
        W = Rand(k, k, cin, cout, device=device)
        W.requires_grad = True
        y = kim.conv(X, W, stride=stride, padding=padding).sum()
        y.backward()
        if wrtX:
            return W.grad
        else:
            return X.grad

    def DoConvLayerBackward(batches, cin, cout, n, k=3, stride=1, bias=True, device=kim.cpu(), wrtX=True):
        X = Rand(batches, cin, n, n, device=device)
        X.requires_grad = True
        f = kim.nn.Conv(cin, cout, k, stride=stride, bias=bias, device=device)
        y = f(X).sum()
        y.backward()
        if wrtX:
            return f.weight.grad
        else:
            return X.grad

    MugradeSubmit(DoConvOpBackward(2, 1, 2, 4, k=1, stride=1, padding=0, wrtX=True))
    MugradeSubmit(DoConvOpBackward(2, 3, 1, 6, k=1, stride=2, padding=0, wrtX=True))
    MugradeSubmit(DoConvOpBackward(2, 1, 2, 10, k=3, stride=1, padding=1, wrtX=True))
    MugradeSubmit(DoConvOpBackward(2, 3, 1, 8, k=3, stride=2, padding=2, wrtX=True))
    MugradeSubmit(DoConvOpBackward(2, 1, 3, 8, k=5, stride=1, padding=2, wrtX=True))

    MugradeSubmit(DoConvOpBackward(2, 1, 2, 4, k=1, stride=1, padding=0, wrtX=False))
    MugradeSubmit(DoConvOpBackward(2, 3, 1, 6, k=1, stride=2, padding=0, wrtX=False))
    MugradeSubmit(DoConvOpBackward(2, 1, 2, 6, k=3, stride=1, padding=1, wrtX=False))
    MugradeSubmit(DoConvOpBackward(2, 3, 1, 6, k=3, stride=2, padding=2, wrtX=False))
    MugradeSubmit(DoConvOpBackward(2, 1, 3, 8, k=5, stride=1, padding=2, wrtX=False))

    MugradeSubmit(DoConvLayerBackward(3, 2, 4, 6, k=3, stride=1, bias=True, wrtX=True))
    MugradeSubmit(DoConvLayerBackward(1, 2, 1, 12, k=7, stride=1, bias=False, wrtX=True))
    MugradeSubmit(DoConvLayerBackward(1, 1, 3, 12, k=7, stride=4, bias=False, wrtX=True))
    MugradeSubmit(DoConvLayerBackward(3, 2, 4, 6, k=3, stride=1, bias=True, wrtX=False))
    MugradeSubmit(DoConvLayerBackward(1, 2, 1, 12, k=7, stride=1, bias=False, wrtX=False))
    MugradeSubmit(DoConvLayerBackward(1, 1, 3, 12, k=7, stride=4, bias=False, wrtX=False))

    if kim.cuda().enabled():
        MugradeSubmit(DoConvLayerBackward(3, 2, 4, 6, k=3, stride=1, bias=False, wrtX=True, device=kim.cuda()))
        MugradeSubmit(DoConvLayerBackward(3, 4, 2, 6, k=3, stride=1, bias=False, wrtX=False, device=kim.cuda()))
    else:
        print('You need a GPU to run these tests!')


def submit_new_ops():
    # pad
    np.random.seed(1337)
    _A = np.random.randint(low=1, high=10, size=(2, 2, 2, 2))
    A  = nd.NDArray(_A, device=nd.cpu())
    MugradeSubmit(A.pad(( (0, 0), (1, 1), (2, 2), (0, 0))))

    def DoFlip(shape, axes, backward=False, device=kim.cpu()):
        X = Rand(*shape, device=device)
        X.requires_grad = True
        Y = kim.flip(X, axes=axes)
        if backward:
            V = Rand(*shape, device=device, entropy=2)
            Z = (V*Y).sum()
            Z.backward()
            return X.grad
        else:
            return Y

    def DoDilate(shape, axes, dilation, backward=False, device=kim.cpu()):
        X = Rand(*shape, device=device)
        X.requires_grad = True
        Y = kim.dilate(X, dilation=dilation, axes=axes)
        if backward:
            V = Rand(*Y.shape, device=device, entropy=2)
            Z = (V*Y).sum()
            Z.backward()
            return X.grad
        else:
            return Y

    # flip
    MugradeSubmit(DoFlip((2, 2, 3, 1), (1,2)))
    MugradeSubmit(DoFlip((2, 1, 3, 2), (0,1,2,3)))
    MugradeSubmit(DoFlip((8, 4), (1,)))
    MugradeSubmit(DoFlip((4, 8), (0,)))
    MugradeSubmit(DoFlip((2, 2, 3, 1), (2,3), backward=True))
    MugradeSubmit(DoFlip((2, 1, 3, 2), (1,2,3), backward=True))

    # dilate
    MugradeSubmit(DoDilate((2, 2, 3, 1), (1,2), 1))
    MugradeSubmit(DoDilate((2, 2), (2,), 1))
    MugradeSubmit(DoDilate((2, 2, 3, 1), (1,2), 1, backward=True))
    MugradeSubmit(DoDilate((2, 2), (2,), 1, backward=True))



def submit_resnet9():
    def num_params(model):
        return np.sum([np.prod(x.shape) for x in model.parameters()])

    device = kim.cpu()
    import sys
    sys.path.append('.')
    from apps.models import ResNet9
    np.random.seed(1)
    model = ResNet9(device=device)

    MugradeSubmit(kim.Tensor(num_params(model)))

    np.random.seed(1)
    dataset = kim.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
    dataloader = kim.data.DataLoader(\
             dataset=dataset,
             batch_size=128,
             shuffle=True
             )
    np.random.seed(1)
    model = ResNet9(device=device, dtype="float32")
    out = one_iter_of_cifar10_training(dataloader, model, niter=2, opt=kim.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001), device=device)
    MugradeSubmit(kim.Tensor(list(out)))


if __name__ == "__main__":
    submit_conv_forward()
    submit_conv_backward()
    submit_new_ops()
    submit_resnet9()
