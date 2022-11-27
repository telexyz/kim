import numpy as np
import kim
import kim.nn as nn
from kim import as_numpy

def get_tensor(*shape, entropy=1):
    np.random.seed(np.prod(shape) * len(shape) * entropy)
    return kim.Tensor(np.random.randint(0, 100, size=shape) / 20, dtype="float32")

def get_int_tensor(*shape, low=0, high=10, entropy=1):
    np.random.seed(np.prod(shape) * len(shape) * entropy)
    return kim.Tensor(np.random.randint(low, high, size=shape))

def global_tensor_count():
    return np.array(kim.autograd.CompGraph.NODE_COUNT)

def learn_model_1d(feature_size, nclasses, _model, optimizer, epochs=1, **kwargs):
    np.random.seed(42)
    model = _model([])
    X = get_tensor(1024, feature_size).cached_data
    y = get_int_tensor(1024, low=0, high=nclasses).cached_data
    # y = y.astype(np.uint8)
    m = X.shape[0]
    batch = 32

    loss_func = nn.SoftmaxLoss()
    opt = optimizer(model.parameters(), **kwargs)
    
    X = as_numpy(X)
    y = as_numpy(y)
    for _ in range(epochs):
        for i, (X0, y0) in enumerate(zip(np.array_split(X, m//batch), np.array_split(y, m//batch))):
            opt.reset_grad()
            X0, y0 = kim.Tensor(X0, dtype="float32"), kim.Tensor(y0)
            out = model(X0)
            loss = loss_func(out, y0)
            assert loss.dtype == "float32"
            # print(">>> loss", loss, loss.dtype) # => float64
            loss.backward()
            # Opt should not change gradients.
            grad_before = model.parameters()[0].grad.detach().cached_data
            opt.step()
            grad_after = model.parameters()[0].grad.detach().cached_data
            # print(">>>", grad_before, grad_after)
            np.testing.assert_allclose(as_numpy(grad_before), as_numpy(grad_after), rtol=1e-5, atol=1e-5, err_msg="Optim should not modify gradients in place")
    return np.array(loss.cached_data)

def learn_model_1d_eval(feature_size, nclasses, _model, optimizer, epochs=1, **kwargs):
    np.random.seed(42)
    model = _model([])
    X = as_numpy(get_tensor(1024, feature_size).cached_data)
    y = get_int_tensor(1024, low=0, high=nclasses).cached_data
    y = as_numpy(y).astype(np.uint8)
    m = X.shape[0]
    batch = 32

    loss_func = nn.SoftmaxLoss()
    opt = optimizer(model.parameters(), **kwargs)

    for i, (X0, y0) in enumerate(zip(np.array_split(X, m//batch), np.array_split(y, m//batch))):
        opt.reset_grad()
        X0, y0 = kim.Tensor(X0, dtype="float32"), kim.Tensor(y0)
        out = model(X0)
        loss = loss_func(out, y0)
        loss.backward()
        opt.step()

    X_test = kim.Tensor(get_tensor(batch, feature_size).cached_data)
    y_test = kim.Tensor(get_int_tensor(batch, low=0, high=nclasses).cached_data)
    y_test = as_numpy(y_test).astype(np.uint8)

    model.eval()

    return np.array(loss_func(model(X_test), y_test).cached_data)


def assert_allclose(x, y, rtol=1e-5, atol=1e-5):
    if kim.array_api == np:
        np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-5)
    else:
        if kim.default_device() == kim.cuda_triton(): 
            rtol = 1e-2 # reduce accuracy, since triton use f16 matmul
        # work around since assert_allclose cause error on nd backend
        xy = np.absolute(as_numpy((x-y).sum()))
        assert abs(xy) < rtol

def test_optim_sgd_vanilla_1():
    assert_allclose(learn_model_1d(64, 16, lambda z: nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16)), kim.optim.SGD, lr=0.01, momentum=0.0), np.array(3.207009), rtol=1e-5, atol=1e-5)

def test_optim_sgd_momentum_1():
    assert_allclose(learn_model_1d(64, 16, lambda z: nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16)), kim.optim.SGD, lr=0.01, momentum=0.9), np.array(3.311805), rtol=1e-5, atol=1e-5)

def test_optim_sgd_weight_decay_1():
    assert_allclose(learn_model_1d(64, 16, lambda z: nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16)), kim.optim.SGD, lr=0.01, momentum=0.0, weight_decay=0.01), np.array(3.202637), rtol=1e-5, atol=1e-5)

def test_optim_sgd_momentum_weight_decay_1():
    assert_allclose(learn_model_1d(64, 16, lambda z: nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16)), kim.optim.SGD, lr=0.01, momentum=0.9, weight_decay=0.01),
        np.array(3.306993), rtol=1e-5, atol=1e-5)

def test_optim_sgd_layernorm_residual_1():
    nn.LayerNorm1d(8)
    assert_allclose(learn_model_1d(64, 16, lambda z: nn.Sequential(nn.Linear(64, 8), nn.ReLU(), nn.Residual(nn.Linear(8, 8)), nn.Linear(8, 16)), kim.optim.SGD, epochs=3, lr=0.01, weight_decay=0.001),
        np.array(2.852236), rtol=1e-5, atol=1e-5)

# We're checking that you have not allocated too many tensors;
# if this fails, make sure you're using .detach()/.data whenever possible.
def test_optim_sgd_z_memory_check_1():
    # np.testing.assert_allclose
    np.testing.assert_allclose(global_tensor_count(),
        np.array(387//2), rtol=1e-5, atol=387//2)


def test_optim_adam_1():
    assert_allclose(learn_model_1d(64, 16, lambda z: nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16)), kim.optim.Adam, lr=0.001),
        np.array(3.703999), rtol=1e-5, atol=1e-5)

def test_optim_adam_weight_decay_1():
    assert_allclose(learn_model_1d(64, 16, lambda z: nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16)), kim.optim.Adam, lr=0.001, weight_decay=0.01),
        np.array(3.705134), rtol=1e-5, atol=1e-5)

def test_optim_adam_batchnorm_1():
    assert_allclose(learn_model_1d(64, 16, lambda z: nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Linear(32, 16)), kim.optim.Adam, lr=0.001, weight_decay=0.001),
        np.array(3.296256, dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_optim_adam_batchnorm_eval_mode_1():
    assert_allclose(learn_model_1d_eval(64, 16, lambda z: nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Linear(32, 16)), kim.optim.Adam, lr=0.001, weight_decay=0.001),
        np.array(3.192054, dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_optim_adam_layernorm_1():
    assert_allclose(learn_model_1d(64, 16, lambda z: nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.LayerNorm1d(32), nn.Linear(32, 16)), kim.optim.Adam, lr=0.01, weight_decay=0.01),
        np.array(2.82192, dtype=np.float32), rtol=1e-5, atol=1e-5)

def test_optim_adam_weight_decay_bias_correction_1():
    assert_allclose(learn_model_1d(64, 16, lambda z: nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16)), kim.optim.Adam, lr=0.001, weight_decay=0.01),
        np.array(3.705134), rtol=1e-5, atol=1e-5)

# We're checking that you have not allocated too many tensors;
# if this fails, make sure you're using .detach()/.data whenever possible.
def test_optim_adam_z_memory_check_1():
    np.testing.assert_allclose(global_tensor_count(),
        np.array(1000//2), rtol=1e-5, atol=1000//2)
