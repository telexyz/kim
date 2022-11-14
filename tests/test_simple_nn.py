import numpy as np
import numdifftools as nd
import kim

from gradient_check import *

import sys
sys.path.append('./apps')
from simple_nn import *

##############################################################################
### TESTS/SUBMISSION CODE FOR softmax_loss

def test_nn_softmax_loss():
    X,y = parse_mnist("data/train-images-idx3-ubyte.gz",
                      "data/train-labels-idx1-ubyte.gz")
    np.random.seed(0)
    Z = kim.Tensor(np.zeros((y.shape[0], 10)).astype(np.float32))
    y_one_hot = np.zeros((y.shape[0], 10))
    y_one_hot[np.arange(y.size), y] = 1
    y = kim.Tensor(y_one_hot)
    np.testing.assert_allclose(softmax_loss(Z,y).numpy(), 2.3025850, rtol=1e-3, atol=1e-3)
    Z = kim.Tensor(np.random.randn(y.shape[0], 10).astype(np.float32))
    np.testing.assert_allclose(softmax_loss(Z,y).numpy(), 2.7291998, rtol=1e-6, atol=1e-6)

    # test softmax loss backward
    Zsmall = kim.Tensor(np.random.randn(16, 10).astype(np.float32))
    ysmall = kim.Tensor(y_one_hot[:16])
    gradient_check(softmax_loss, Zsmall, ysmall, tol=0.01, backward=True)


##############################################################################
### TESTS/SUBMISSION CODE FOR nn_epoch

def test_nn_epoch():
    # test nn gradients
    np.random.seed(0)
    X = np.random.randn(50,5).astype(np.float32)
    y = np.random.randint(3, size=(50,)).astype(np.uint8)
    W1 = np.random.randn(5, 10).astype(np.float32) / np.sqrt(10)
    W2 = np.random.randn(10, 3).astype(np.float32) / np.sqrt(3)
    W1_0, W2_0 = W1.copy(), W2.copy()
    W1 = kim.Tensor(W1)
    W2 = kim.Tensor(W2)
    X_ = kim.Tensor(X)
    y_one_hot = np.zeros((y.shape[0], 3))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = kim.Tensor(y_one_hot)
    dW1 = nd.Gradient(lambda W1_ :
        softmax_loss(kim.relu(X_@kim.Tensor(W1_).reshape((5,10)))@W2, y_).numpy())(W1.numpy())
    dW2 = nd.Gradient(lambda W2_ :
        softmax_loss(kim.relu(X_@W1)@kim.Tensor(W2_).reshape((10,3)), y_).numpy())(W2.numpy())
    W1, W2 = nn_epoch(X, y, W1, W2, lr=1.0, batch=50)
    np.testing.assert_allclose(dW1.reshape(5,10), W1_0-W1.numpy(), rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(dW2.reshape(10,3), W2_0-W2.numpy(), rtol=1e-4, atol=1e-4)

    # test full epoch
    X,y = parse_mnist("data/train-images-idx3-ubyte.gz",
                      "data/train-labels-idx1-ubyte.gz")
    np.random.seed(0)
    W1 = kim.Tensor(np.random.randn(X.shape[1], 100).astype(np.float32) / np.sqrt(100))
    W2 = kim.Tensor(np.random.randn(100, 10).astype(np.float32) / np.sqrt(10))
    W1, W2 = nn_epoch(X, y, W1, W2, lr=0.2, batch=100)
    np.testing.assert_allclose(np.linalg.norm(W1.numpy()), 28.437788,
                               rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(W2.numpy()), 10.455095,
                               rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(loss_err(kim.relu(kim.Tensor(X)@W1)@W2, y),
                               (0.19770025, 0.06006667), rtol=1e-4, atol=1e-4)
