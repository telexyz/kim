import sys
sys.path.append('..')
import kim
import kim.nn as nn
from kim import backend_ndarray as nd
from models import *

from timeit import default_timer as timer
import datetime

device = kim.default_device()

### CIFAR-10 training ###

def epoch_general_cifar10(dataloader, model, started_at, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    training = (opt is not None)
    if training: model.train()
    else: model.eval()

    correct, total_loss = 0, 0
    n = 0; niter = 0
    for (X, y) in dataloader:
        out = model(X)
        loss = loss_fn(out, y)
        correct += np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())
        total_loss += loss.data.numpy()
        if training:
            opt.reset_grad()
            loss.backward()
            opt.step()
        niter += 1; n += y.shape[0] # n += batch_size
        if niter % 20 == 0:
            time_passed = datetime.timedelta(seconds=timer() - started_at)
            print("iter: %s, acc: %.5f, loss: %.5f (%s)" % (niter, correct/n, total_loss/niter, time_passed))

    return correct/n, total_loss/niter


def train_cifar10(model, dataloader, n_epochs=1, optimizer=kim.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay, device=device)
    lf = loss_fn()
    started_at = timer()
    for epoch in range(n_epochs):
        avg_acc, avg_loss = epoch_general_cifar10(dataloader, model, started_at, loss_fn=lf, opt=opt)
        time_passed = datetime.timedelta(seconds=timer() - started_at)
        print("\n>>> Training epoch: %s, acc: %s, loss: %s (%s)\n" % (epoch, avg_acc, avg_loss, time_passed))
    return avg_acc, avg_loss

def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    lf = loss_fn()
    started_at = timer()
    avg_acc, avg_loss = epoch_general_cifar10(dataloader, model, started_at, loss_fn=lf, opt=None)
    time_passed = datetime.timedelta(seconds=timer() - started_at)
    print("\n>>> Test acc: %s, loss: %s (%s)\n" % (avg_acc, avg_loss, time_passed))
    return avg_acc, avg_loss


### PTB training ###
def epoch_general_ptb(data, model, started_at, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    training = (opt is not None)
    if training: model.train()
    else: model.eval()

    correct, total_loss = 0, 0
    n = 0; niter = 0
    nbatch, batch_size = data.shape
    # print(">>>", data.shape, seq_len)
    # assert False
    for i in range(nbatch - seq_len):
        X, y = kim.data.get_batch(data, i, seq_len)
        out = model(X)[0]
        loss = loss_fn(out, y)
        correct += np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())
        total_loss += loss.numpy()
        if training:
            opt.reset_grad()
            loss.backward()
            opt.step()
        niter += 1; n += batch_size
        if niter % 20 == 0:
            time_passed = datetime.timedelta(seconds=timer() - started_at)
            print("iter: %s, acc: %.5f, loss: %.5f (%s)" % (niter, correct/n, total_loss/niter, time_passed))

    return correct/n, total_loss/niter
    ### END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=kim.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay, device=device)
    lf = loss_fn()
    started_at = timer()
    for epoch in range(n_epochs):
        avg_acc, avg_loss = epoch_general_ptb(data, model, started_at, loss_fn=lf, opt=opt, seq_len=seq_len)
        time_passed = datetime.timedelta(seconds=timer() - started_at)
        print("\n>>> Training epoch: %s, acc: %s, loss: %s (%s)\n" % (epoch, avg_acc, avg_loss, time_passed))
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    lf = loss_fn()
    started_at = timer()
    avg_acc, avg_loss = epoch_general_ptb(data, model, started_at, loss_fn=lf, opt=None, seq_len=seq_len)
    time_passed = datetime.timedelta(seconds=timer() - started_at)
    print("\n>>> Test acc: %s, loss: %s (%s)\n" % (avg_acc, avg_loss, time_passed))
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    ### For testing purposes
    device = None
    #dataset = kim.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
    #dataloader = kim.data.DataLoader(\
    #         dataset=dataset,
    #         batch_size=128,
    #         shuffle=True
    #         )
    #
    #model = ResNet9(device=device, dtype="float32")
    #train_cifar10(model, dataloader, n_epochs=10, optimizer=kim.optim.Adam,
    #      lr=0.001, weight_decay=0.001)

    corpus = kim.data.Corpus("../data/ptb")
    seq_len = 40 # ko được thay đổi, nếu ko loss sẽ tăng
    batch_size = 16 # ko được thay đổi, nếu ko loss sẽ tăng
    hidden_size = 128

    train_data = kim.data.batchify(corpus.train, batch_size, device=device, dtype="float32")
    print("Train data:", train_data.shape)
    print("seq, batch, hidden:", seq_len, batch_size, hidden_size)
    model = LanguageModel(1, len(corpus.dictionary), hidden_size, num_layers=2, device=device)
    train_ptb(model, train_data, seq_len, n_epochs=10, device=device)

    test_data = kim.data.batchify(corpus.test, batch_size, device=device, dtype="float32")
    evaluate_ptb(model, test_data, seq_len, device=device)