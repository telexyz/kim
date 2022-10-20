import numpy as np
import kim as ndl
import kim.nn as nn

import sys
sys.path.append("./apps")
from mlp_resnet import *

def num_params(model):
    return np.sum([np.prod(x.shape) for x in model.parameters()])

def residual_block_num_params(dim, hidden_dim, norm):
    model = ResidualBlock(dim, hidden_dim, norm)
    return np.array(num_params(model))

def residual_block_forward(dim, hidden_dim, norm, drop_prob):
    np.random.seed(2)
    input_tensor = ndl.Tensor(np.random.randn(1, dim))
    output_tensor = ResidualBlock(dim, hidden_dim, norm, drop_prob)(input_tensor)
    return output_tensor.numpy()


def mlp_resnet_num_params(dim, hidden_dim, num_blocks, num_classes, norm):
    model = MLPResNet(dim, hidden_dim, num_blocks, num_classes, norm)
    return np.array(num_params(model))

def mlp_resnet_forward(dim, hidden_dim, num_blocks, num_classes, norm, drop_prob):
    np.random.seed(4)
    input_tensor = ndl.Tensor(np.random.randn(2, dim), dtype=np.float32)
    output_tensor = MLPResNet(dim, hidden_dim, num_blocks, num_classes, norm, drop_prob)(input_tensor)
    return output_tensor.numpy()

def train_epoch_1(hidden_dim, batch_size, optimizer, **kwargs):
    np.random.seed(1)
    train_dataset = ndl.data.MNISTDataset(\
            "./data/train-images-idx3-ubyte.gz",
            "./data/train-labels-idx1-ubyte.gz")
    train_dataloader = ndl.data.DataLoader(\
             dataset=train_dataset,
             batch_size=batch_size)

    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), **kwargs)
    model.eval()
    return np.array(epoch(train_dataloader, model, opt))

def eval_epoch_1(hidden_dim, batch_size):
    np.random.seed(1)
    test_dataset = ndl.data.MNISTDataset(\
            "./data/t10k-images-idx3-ubyte.gz",
            "./data/t10k-labels-idx1-ubyte.gz")
    test_dataloader = ndl.data.DataLoader(\
             dataset=test_dataset,
             batch_size=batch_size,
             shuffle=False)

    model = MLPResNet(784, hidden_dim)
    model.train()
    return np.array(epoch(test_dataloader, model))

def train_mnist_1(batch_size, epochs, optimizer, lr, weight_decay, hidden_dim):
    np.random.seed(1)
    out = train_mnist(batch_size, epochs, optimizer, lr, weight_decay, hidden_dim, data_dir="./data")
    return np.array(out)


def test_mlp_residual_block_num_params_1():
    np.testing.assert_allclose(residual_block_num_params(15, 2, nn.BatchNorm1d),
        np.array(111), rtol=1e-5, atol=1e-5)

def test_mlp_residual_block_num_params_2():
    np.testing.assert_allclose(residual_block_num_params(784, 100, nn.LayerNorm1d),
        np.array(159452), rtol=1e-5, atol=1e-5)

def test_mlp_residual_block_forward_1():
    np.testing.assert_allclose(
        residual_block_forward(15, 10, nn.LayerNorm1d, 0.5),
        np.array([[
            0., 1.358399, 0., 1.384224, 0., 0., 0.255451, 0.077662, 0.,
            0.939582, 0.525591, 1.99213, 0., 0., 1.012827
        ]],
        dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )

def test_mlp_resnet_num_params_1():
    np.testing.assert_allclose(mlp_resnet_num_params(150, 100, 5, 10, nn.LayerNorm1d),
        np.array(68360), rtol=1e-5, atol=1e-5)

def test_mlp_resnet_num_params_2():
    np.testing.assert_allclose(mlp_resnet_num_params(10, 100, 1, 100, nn.BatchNorm1d),
        np.array(21650), rtol=1e-5, atol=1e-5)

def test_mlp_resnet_forward_1():
    np.testing.assert_allclose(
        mlp_resnet_forward(10, 5, 2, 5, nn.LayerNorm1d, 0.5),
        np.array([[3.046162, 1.44972, -1.921363, 0.021816, -0.433953],
                  [3.489114, 1.820994, -2.111306, 0.226388, -1.029428]],
                 dtype=np.float32),
        rtol=1e-5,
        atol=1e-5)

def test_mlp_resnet_forward_2():
    np.testing.assert_allclose(
        mlp_resnet_forward(15, 25, 5, 14, nn.BatchNorm1d, 0.0),
        np.array([[
            0.92448235, -2.745743, -1.5077105, 1.130784, -1.2078242,
            -0.09833566, -0.69301605, 2.8945382, 1.259397, 0.13866742,
            -2.963875, -4.8566914, 1.7062538, -4.846424
        ],
        [
            0.6653336, -2.4708004, 2.0572243, -1.0791507, 4.3489094,
            3.1086435, 0.0304327, -1.9227124, -1.416201, -7.2151937,
            -1.4858506, 7.1039696, -2.1589825, -0.7593413
        ]],
        dtype=np.float32),
        rtol=1e-5,
        atol=1e-5)

def test_mlp_train_epoch_1():
    np.testing.assert_allclose(train_epoch_1(5, 250, ndl.optim.Adam, lr=0.01, weight_decay=0.1),
        np.array([0.675267, 1.84043]), rtol=0.0001, atol=0.0001)

def test_mlp_eval_epoch_1():
    np.testing.assert_allclose(eval_epoch_1(10, 150),
        np.array([0.9164 , 4.137814]), rtol=1e-5, atol=1e-5)

def test_mlp_train_mnist_1():
    np.testing.assert_allclose(train_mnist_1(250, 2, ndl.optim.SGD, 0.001, 0.01, 100),
        np.array([0.4875 , 1.462595, 0.3245 , 1.049429]), rtol=0.001, atol=0.001)
