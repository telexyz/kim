import kim as ndl
import mugrade
from gradient_check import *

import sys
sys.path.append('./apps')
from simple_nn import *

def submit_forward():
    mugrade.submit(ndl.divide(ndl.Tensor([[3.4 , 2.35, 1.25 ], [0.45, 1.95, 2.55]]),
                              ndl.Tensor([[4.9 , 4.35, 4.1 ], [0.65, 0.7 , 4.04 ]])).numpy())
    mugrade.submit(ndl.divide_scalar(ndl.Tensor([[1.4 , 2.89]]), scalar=7).numpy())
    mugrade.submit(ndl.matmul(ndl.Tensor([[1.75, 1.75, 0.25], [4.95, 4.35, 0.3], [0.3, 1.4, 2.1]]),
                              ndl.Tensor([[2.35, 2.2, 1.85], [7.85, 4.88, 2.6], [1.15, 0.25, 4.19]])).numpy())
    mugrade.submit(ndl.summation(ndl.Tensor([[1.2, 4.35, 1.4, 0.3, 0.75],
                                             [2., 1.85, 7.75, 3.7, 1.55],
                                             [9.2, 2.3, 3.45, 0.7, 0.]])).numpy())
    mugrade.submit(ndl.summation(ndl.Tensor([[5.05, 2.55, 1.],
                                             [2.75, 3.7, 2.1],
                                             [0.1, 4.1, 3.3],
                                             [1.4, 0.4, 3.4],
                                             [2.8, 0.55, 2.9]]), axes=1).numpy())
    mugrade.submit(ndl.broadcast_to(ndl.Tensor([[1.95, 3.85, -0.6]]), shape=(3, 3, 3)).numpy())
    mugrade.submit(ndl.reshape(ndl.Tensor([[7.9, 2., 2.4],
                                           [3.11, 3.95, 0.65],
                                           [2.1, 2.18, 2.2],
                                           [1.9, 4.54, 3.25],
                                           [1.35, 7.45, 3.45]]), shape=(15,)).numpy())
    mugrade.submit(ndl.reshape(ndl.Tensor([[[5.1, 4.05, 1.25, 4.65],
                                            [3.65, 0.9, 0.65, 1.65]],
                                           [[4.7, 1.4, 2.55, 4.8],
                                            [2.8, 1.75, 3.8, 0.6]],
                                           [[3.75, 0.6, 1., 3.5],
                                            [8.15, 1.9, 4.55, 2.83]]]), shape=(2, 3, 4)).numpy())
    mugrade.submit(ndl.negate(ndl.Tensor([[1.45, 0.55]])).numpy())
    mugrade.submit(ndl.transpose(ndl.Tensor([[[3.45]],
                                             [[2.54]],
                                             [[1.91]]]), axes=(0, 1)).numpy())
    mugrade.submit(ndl.transpose(ndl.Tensor([[4.45, 2.15],
                                             [1.89, 1.21],
                                             [6.15, 2.42]])).numpy())


def submit_backward():
    np.random.seed(0)
    out = gradient_check(ndl.divide, ndl.Tensor(np.random.randn(3, 5)), ndl.Tensor(6 + np.random.randn(3, 5)))
    # print(out)
    mugrade.submit(out)
    mugrade.submit(gradient_check(ndl.divide_scalar, ndl.Tensor(np.random.randn(3, 5)), scalar=np.random.randn(1)))
    mugrade.submit(gradient_check(ndl.matmul, ndl.Tensor(np.random.randn(1, 5)), ndl.Tensor(np.random.randn(5, 1))))
    mugrade.submit(gradient_check(ndl.matmul, ndl.Tensor(np.random.randn(2, 4)), ndl.Tensor(np.random.randn(4, 2))))
    mugrade.submit(gradient_check(ndl.matmul, ndl.Tensor(np.random.randn(2, 4)), ndl.Tensor(np.random.randn(7, 4, 2))))
    mugrade.submit(gradient_check(ndl.matmul, ndl.Tensor(np.random.randn(3, 2, 1)), ndl.Tensor(np.random.randn(3, 3, 1, 2))))
    mugrade.submit(gradient_check(ndl.matmul, ndl.Tensor(np.random.randn(2, 4)), ndl.Tensor(np.random.randn(2, 4, 4, 2))))
    mugrade.submit(gradient_check(ndl.reshape, ndl.Tensor(np.random.randn(5, 4)), shape=(5,4,1)))
    mugrade.submit(gradient_check(ndl.reshape, ndl.Tensor(np.random.randn(5, 4)), shape=(2, 2, 5)))
    mugrade.submit(gradient_check(ndl.negate, ndl.Tensor(np.random.randn(1, 4, 2))))
    mugrade.submit(gradient_check(ndl.transpose, ndl.Tensor(np.random.randn(3, 2, 4)), axes=(0, 2)))
    mugrade.submit(gradient_check(ndl.broadcast_to, ndl.Tensor(np.random.randn(7, 1)), shape=(7, 7)))
    mugrade.submit(gradient_check(ndl.broadcast_to, ndl.Tensor(np.random.randn(1, 5)), shape=(5, 5)))
    mugrade.submit(gradient_check(ndl.broadcast_to, ndl.Tensor(np.random.randn(1,)), shape=(4, 4, 4)))
    mugrade.submit(gradient_check(ndl.broadcast_to, ndl.Tensor(np.random.randn()), shape=(1, 3, 6)))
    mugrade.submit(gradient_check(ndl.broadcast_to, ndl.Tensor(np.random.randn(4,4,1)), shape=(4,4,6)))
    mugrade.submit(gradient_check(ndl.summation, ndl.Tensor(np.random.randn(3,2,1))))
    mugrade.submit(gradient_check(ndl.summation, ndl.Tensor(np.random.randn(3,6)), axes=(1,)))
    mugrade.submit(gradient_check(ndl.summation, ndl.Tensor(np.random.randn(7,)), axes=(0,)))
    mugrade.submit(gradient_check(ndl.summation, ndl.Tensor(np.random.randn(7,8)), axes=(0,1)))
    mugrade.submit(gradient_check(ndl.summation, ndl.Tensor(np.random.randn(5,4,5)), axes=(0,1,2)))


def submit_topo_sort():

    # mugrade test case 1
    a2, b2 = ndl.Tensor(np.asarray([[0.74683138]])), ndl.Tensor(np.asarray([[0.65539231]]))
    c2 = 9 * a2 * a2 + 15 * b2 * a2 - b2

    topo_order = np.array([x.numpy() for x in ndl.autograd.find_topo_sort([c2])])

    mugrade.submit(topo_order)


    # mugrade test case 2
    a1, b1 = ndl.Tensor(np.asarray([[0.9067453], [0.18521121]])), ndl.Tensor(np.asarray([[0.80992494, 0.52458167]]))
    c1 = 3 * ((b1 @ a1) + (2.3412 * b1) @ a1) + 1.5

    topo_order2 = [x.numpy() for x in ndl.autograd.find_topo_sort([c1])]

    mugrade.submit(topo_order2)

    # mugrade test case 3
    c = ndl.Tensor(np.asarray([[-0.16541387, 2.52604789], [-0.31008569, -0.4748876]]))
    d = ndl.Tensor(np.asarray([[0.55936155, -2.12630983], [0.59930618, -0.19554253]]))
    f = (c + d@d - d) @ c

    topo_order3 = np.array([x.numpy() for x in ndl.autograd.find_topo_sort([f])])

    mugrade.submit(topo_order3)


def submit_compute_gradient():
    a = ndl.Tensor(np.array([[-0.2985143, 0.36875625],
                             [-0.918687, 0.52262925]]))
    b = ndl.Tensor(np.array([[-1.58839928, 1.58592338],
                             [-0.15932137, -0.55618462]]))
    c = ndl.Tensor(np.array([[-0.5096208, 0.73466865],
                             [0.38762148, -0.41149092]]))
    d = (a + b)@c@(a + c)
    d.backward()
    grads = [x.grad.numpy() for x in [a, b, c]]
    mugrade.submit(grads)

    # just need a fixed function or two to send results to mugrade
    a = ndl.Tensor(np.array([[0.4736625, 0.06895066, 1.36455087, -0.31170743, 0.1370395],
                             [0.2283258, 0.72298311, -1.20394586, -1.95844434, -0.69535299],
                             [0.18016408, 0.0266557, 0.80940201, -0.45913678, -0.05886218],
                             [-0.50678819, -1.53276348, -0.27915708, -0.571393, -0.17145921]]))
    b = ndl.Tensor(np.array([[0.28738358, -1.27265428, 0.32388374],
                             [-0.77830395, 2.07830592, 0.99796268],
                             [-0.76966429, -1.37012833, -0.16733693],
                             [-0.44134101, -1.24495901, -1.62953897],
                             [-0.75627713, -0.80006226, 0.03875171]]))
    c = ndl.Tensor(np.array([[1.25727301, 0.39400789, 1.29139323, -0.950472],
                             [-0.21250305, -0.93591609, 1.6802009, -0.39765765],
                             [-0.16926597, -0.45218718, 0.38103032, -0.11321965]]))
    output = ndl.summation((a@b)@c@a)
    output.backward()
    grads = [x.grad.numpy() for x in [a, b, c]]
    mugrade.submit(grads)

    x2 = ndl.Tensor(3)
    x3 = ndl.Tensor(2)
    y = x2 * x2 - x2 * x3
    y.backward()
    grad_x2 = x2.grad
    grad_x3 = x3.grad
    # gradient of gradient
    grad_x2.backward()
    grad_x2_x2 = x2.grad
    grad_x2_x3 = x3.grad
    x2_val = x2.numpy()
    x3_val = x3.numpy()
    mugrade.submit(y.numpy())
    mugrade.submit(grad_x2.numpy())
    mugrade.submit(grad_x3.numpy())
    mugrade.submit(grad_x2_x2.numpy())


def submit_softmax_loss_ndl():
    # add a mugrade submit for log
    np.random.seed(0)
    mugrade.submit(gradient_check(ndl.log, ndl.Tensor(1 + np.random.rand(5, 4))))

    X,y = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                      "data/t10k-labels-idx1-ubyte.gz")

    y_one_hot = np.zeros((y.shape[0], 10))
    y_one_hot[np.arange(y.size), y] = 1
    y = ndl.Tensor(y_one_hot)
    mugrade.submit(softmax_loss(ndl.Tensor(np.zeros((y.shape[0], 10)).astype(np.float32)), y).numpy())
    np.random.seed(0)
    mugrade.submit(softmax_loss(ndl.Tensor(np.random.randn(y.shape[0], 10).astype(np.float32)), y).numpy())


def submit_nn_epoch_ndl():
    X,y = parse_mnist("data/train-images-idx3-ubyte.gz",
                      "data/train-labels-idx1-ubyte.gz")
    np.random.seed(1)
    W1 = ndl.Tensor(np.random.randn(X.shape[1], 100).astype(np.float32) / np.sqrt(100))
    W2 = ndl.Tensor(np.random.randn(100, 10).astype(np.float32) / np.sqrt(10))
    W1, W2 = nn_epoch(X[:100], y[:100], W1, W2, lr=0.1, batch=100)

    mugrade.submit(np.linalg.norm(W1.numpy()))
    mugrade.submit(np.linalg.norm(W2.numpy()))

    np.random.seed(1)
    W1 = ndl.Tensor(np.random.randn(X.shape[1], 100).astype(np.float32) / np.sqrt(100))
    W2 = ndl.Tensor(np.random.randn(100, 10).astype(np.float32) / np.sqrt(10))
    W1, W2 = nn_epoch(X, y, W1, W2, lr=0.2, batch=100)

    mugrade.submit(np.linalg.norm(W1.numpy()))
    mugrade.submit(np.linalg.norm(W2.numpy()))
    mugrade.submit(loss_err(ndl.Tensor(np.maximum(X@W1.numpy(),0))@W2, y))
