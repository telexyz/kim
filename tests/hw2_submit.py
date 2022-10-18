import kim as ndl
import numpy as np
import mugrade

from test_ops import *
from test_nn import *
from test_optim import *

import sys
sys.path.append("./apps")
from mlp_resnet import *

def submit_flip_horizontal():
    tform = ndl.data.RandomFlipHorizontal(0.5)
    np.random.seed(0)
    for _ in range(2):
        size_a, size_b, size_c = np.random.randint(1,5), np.random.randint(1,5), np.random.randint(1,5)
        mugrade.submit(tform(np.random.rand(size_a, size_b,size_c)))

    tform = ndl.data.RandomFlipHorizontal(0)
    for _ in range(2):
        size_a, size_b, size_c = np.random.randint(1,5), np.random.randint(1,5), np.random.randint(1,5)
        mugrade.submit(tform(np.random.rand(size_a, size_b,size_c)))

    tform = ndl.data.RandomFlipHorizontal(1.0)
    for _ in range(2):
        size_a, size_b, size_c = np.random.randint(1,5), np.random.randint(1,5), np.random.randint(1,5)
        mugrade.submit(tform(np.random.rand(size_a, size_b,size_c)))



def submit_random_crop():
    np.random.seed(0)
    tform = ndl.data.RandomCrop(0)
    for _ in range(2):
        size_a, size_b, size_c = np.random.randint(4,5), np.random.randint(4,6), np.random.randint(4,7)
        mugrade.submit(tform(np.random.rand(size_a, size_b,size_c)))

    tform = ndl.data.RandomCrop(2)
    for _ in range(2):
        size_a, size_b, size_c = np.random.randint(4,5), np.random.randint(4,6), np.random.randint(4,7)
        mugrade.submit(tform(np.random.rand(size_a, size_b,size_c)))

    tform = ndl.data.RandomCrop(3)
    for _ in range(2):
        ize_a, size_b, size_c = np.random.randint(4,5), np.random.randint(4,6), np.random.randint(4,7)
        mugrade.submit(tform(np.random.rand(size_a, size_b,size_c)))



def submit_mnist_dataset():
    mnist_train_dataset = ndl.data.MNISTDataset("data/train-images-idx3-ubyte.gz",
                                                "data/train-labels-idx1-ubyte.gz")
    mugrade.submit(mnist_train_dataset[69][:25])
    mugrade.submit(len(mnist_train_dataset))
    np.random.seed(0)
    tforms = [ndl.data.RandomFlipHorizontal()]
    mnist_train_dataset = ndl.data.MNISTDataset("data/train-images-idx3-ubyte.gz",
                                                "data/train-labels-idx1-ubyte.gz",
                                                transforms=tforms)

    for i in [822, 69, 420, 96]:
        mugrade.submit(mnist_train_dataset[i][:-25])


    tforms = [ndl.data.RandomCrop(15), ndl.data.RandomFlipHorizontal()]
    mnist_train_dataset = ndl.data.MNISTDataset("data/train-images-idx3-ubyte.gz",
                                                "data/train-labels-idx1-ubyte.gz",
                                                transforms=tforms)

    for i in [822, 69, 420, 96]:
        mugrade.submit(mnist_train_dataset[i][:-25])



def submit_op_logsumexp():
    mugrade.submit(logsumexp_forward((2,2,2), None))
    mugrade.submit(logsumexp_forward((1,2,3), (0,)))
    mugrade.submit(logsumexp_forward((2,3,3),(1,2)))
    mugrade.submit(logsumexp_forward((1,2,2,2,2), (1,2,3,4)))
    mugrade.submit(logsumexp_forward((1,2,2,2,2), (0,1,3)))
    mugrade.submit(logsumexp_backward((2,2,2), None))
    mugrade.submit(logsumexp_backward((1,2,3), (0,)))
    mugrade.submit(logsumexp_backward((2,3,3),(1,2)))
    mugrade.submit(logsumexp_backward((1,2,2,2,2), (1,2,3,4)))
    mugrade.submit(logsumexp_backward((1,2,2,2,2), (0,1,3)))



def submit_init():
    np.random.seed(0)
    mugrade.submit(ndl.init.kaiming_normal(2,5).numpy())
    mugrade.submit(ndl.init.kaiming_uniform(2,5).numpy())
    mugrade.submit(ndl.init.xavier_uniform(2,5, gain=0.33).numpy())
    mugrade.submit(ndl.init.xavier_normal(2,5, gain=1.3).numpy())


def submit_nn_linear():
    mugrade.submit(linear_forward((3, 5), (1, 3)))
    mugrade.submit(linear_forward((3, 5), (3, 3)))
    mugrade.submit(linear_forward((3, 5), (1, 3, 3)))
    mugrade.submit(linear_backward((4, 5), (1, 4)))
    mugrade.submit(linear_backward((4, 5), (3, 4)))
    mugrade.submit(linear_backward((4, 5), (1, 3, 4)))


def submit_nn_relu():
    mugrade.submit(relu_forward(2, 3))
    mugrade.submit(relu_backward(3, 4))


def submit_nn_sequential():
    mugrade.submit(sequential_forward(batches=2))
    mugrade.submit(sequential_backward(batches=2))


def submit_nn_softmax_loss():
    mugrade.submit(softmax_loss_forward(4, 9))
    mugrade.submit(softmax_loss_forward(2, 7))
    mugrade.submit(softmax_loss_backward(4, 9))
    mugrade.submit(softmax_loss_backward(2, 7))


def submit_nn_layernorm():
    mugrade.submit(layernorm_forward((1,1), 1))
    mugrade.submit(layernorm_forward((10,10), 10))
    mugrade.submit(layernorm_forward((10,30), 30))
    mugrade.submit(layernorm_forward((1,3), 3))
    mugrade.submit(layernorm_backward((1,1), 1))
    mugrade.submit(layernorm_backward((10,10), 10))
    mugrade.submit(layernorm_backward((10,30), 30))
    mugrade.submit(layernorm_backward((1,3), 3))


def submit_nn_batchnorm():
    mugrade.submit(batchnorm_forward(2, 3))
    mugrade.submit(batchnorm_forward(3, 4, affine=True))
    mugrade.submit(batchnorm_backward(5, 3))

    # todo(Zico) : these need to be added to mugrade
    mugrade.submit(batchnorm_backward(4, 2, affine=True))
    mugrade.submit(batchnorm_running_mean(3, 3))
    mugrade.submit(batchnorm_running_mean(3, 3))
    mugrade.submit(batchnorm_running_var(4, 3))
    mugrade.submit(batchnorm_running_var(4, 4))
    mugrade.submit(batchnorm_running_grad(4, 3))


def submit_nn_dropout():
    mugrade.submit(dropout_forward((3, 3), prob=0.4))
    mugrade.submit(dropout_backward((3, 3), prob=0.15))


def submit_nn_residual():
    mugrade.submit(residual_forward(shape=(3,4)))
    mugrade.submit(residual_backward(shape=(3,4)))


def submit_nn_flatten():
    mugrade.submit(flatten_forward(1,2,2))
    mugrade.submit(flatten_forward(2,2,2))
    mugrade.submit(flatten_forward(2,3,4,2,1,2))
    mugrade.submit(flatten_forward(2,3))
    mugrade.submit(flatten_backward(1,2,2))
    mugrade.submit(flatten_backward(2,2,2))
    mugrade.submit(flatten_backward(2,3,4,2,1,2))
    mugrade.submit(flatten_backward(2,3,4,4))
    

def submit_optim_sgd():
    mugrade.submit(learn_model_1d(48, 17, lambda z: nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 17)), ndl.optim.SGD, lr=0.03, momentum=0.0, epochs=2))
    mugrade.submit(learn_model_1d(48, 16, lambda z: nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 16)), ndl.optim.SGD, lr=0.01, momentum=0.9, epochs=2))
    mugrade.submit(learn_model_1d(48, 16, lambda z: nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Linear(32, 16)), ndl.optim.SGD, lr=0.01, momentum=0.0, weight_decay=0.01, epochs=2))
    mugrade.submit(learn_model_1d(54, 16, lambda z: nn.Sequential(nn.Linear(54, 32), nn.ReLU(), nn.Linear(32, 16)), ndl.optim.SGD, lr=0.01, momentum=0.9, weight_decay=0.01, epochs=2))
    mugrade.submit(learn_model_1d(64, 4, lambda z: nn.Sequential(nn.Linear(64, 8), nn.ReLU(), nn.Residual(nn.Linear(8, 8)), nn.Linear(8, 4)), ndl.optim.SGD, epochs=3, lr=0.01, weight_decay=0.001))


def submit_optim_adam():
    mugrade.submit(learn_model_1d(48, 16, lambda z: nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 16)), ndl.optim.Adam, lr=0.001, epochs=2))
    mugrade.submit(learn_model_1d(48, 16, lambda z: nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 16)), ndl.optim.Adam, lr=0.001, weight_decay=0.01, epochs=2))
    mugrade.submit(learn_model_1d(48, 16, lambda z: nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Linear(32, 16)), ndl.optim.Adam, lr=0.001, weight_decay=0.001, epochs=3))
    mugrade.submit(learn_model_1d_eval(48, 16, lambda z: nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Linear(32, 16)), ndl.optim.Adam, lr=0.001, weight_decay=0.001,  epochs=2))
    mugrade.submit(learn_model_1d(48, 16, lambda z: nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.LayerNorm1d(32), nn.Linear(32, 16)), ndl.optim.Adam, lr=0.01, weight_decay=0.01,  epochs=2))
    mugrade.submit(learn_model_1d(48, 16, lambda z: nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 16)), ndl.optim.Adam, lr=0.001, weight_decay=0.01,  epochs=2))


def submit_mlp_resnet():
    mugrade.submit(residual_block_num_params(17, 13, nn.BatchNorm1d))
    mugrade.submit(residual_block_num_params(785, 101, nn.LayerNorm1d))
    mugrade.submit(residual_block_forward(15, 5, nn.LayerNorm1d, 0.3))
    mugrade.submit(mlp_resnet_num_params(75, 75, 3, 3, nn.LayerNorm1d))
    mugrade.submit(mlp_resnet_num_params(15, 10, 10, 5, nn.BatchNorm1d))
    mugrade.submit(mlp_resnet_forward(12, 7, 1, 6, nn.LayerNorm1d, 0.8))
    mugrade.submit(mlp_resnet_forward(15, 3, 2, 15, nn.BatchNorm1d, 0.3))
    mugrade.submit(train_epoch_1(7, 256, ndl.optim.Adam, lr=0.01, weight_decay=0.01))
    mugrade.submit(eval_epoch_1(12, 154))
    mugrade.submit(train_mnist_1(554, 1, ndl.optim.SGD, 0.01, 0.01, 7))


def submit_dataloader():
    batch_size = 1
    mnist_train_dataset = ndl.data.MNISTDataset("data/train-images-idx3-ubyte.gz",
                                                "data/train-labels-idx1-ubyte.gz")
    mnist_train_dataloader = ndl.data.DataLoader(dataset=mnist_train_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False)
    subl = []
    for i, batch in enumerate(mnist_train_dataloader):
        batch_x, batch_y = batch[0].numpy(), batch[1].numpy()
        subl.append(np.sum(batch_x[10:15,10:15]))
        subl.append(np.sum(batch_y))
        if i > 2:
            break
    mugrade.submit(subl)

    batch_size = 5
    mnist_test_dataset = ndl.data.MNISTDataset("data/t10k-images-idx3-ubyte.gz",
                                               "data/t10k-labels-idx1-ubyte.gz")
    mnist_test_dataloader = ndl.data.DataLoader(dataset=mnist_test_dataset,
                                                batch_size=batch_size,
                                                shuffle=False)

    subl_x = []
    subl_y = []
    for i, batch in enumerate(mnist_test_dataloader):
        batch_x, batch_y = batch[0].numpy(), batch[1].numpy()
        subl_x.append(batch_x[10:15,10:15])
        subl_y.append(batch_y)

    mugrade.submit(subl_x[-2:])
    mugrade.submit(subl_y[-2:])

    np.random.seed(0)
    shuf = ndl.data.DataLoader(dataset=mnist_test_dataset,
                               batch_size=10,
                               shuffle=True)
    subl_x = []
    subl_y = []
    for i, batch in enumerate(mnist_test_dataloader):
        batch_x, batch_y = batch[0].numpy(), batch[1].numpy()
        subl_x.append(batch_x[10:15,10:15])
        subl_y.append(batch_y)
        if i > 2:
            break
    mugrade.submit(subl_x)
    mugrade.submit(subl_y)
