import sys
sys.path.append('./python')
import itertools
import numpy as np
import pytest
import mugrade

import kim
from kim import backend_ndarray as nd


np.random.seed(2)


_DEVICES = [kim.cpu(), pytest.param(kim.cuda(),
    marks=pytest.mark.skipif(not kim.cuda().enabled(), reason="No GPU"))]


TRAIN = [True, False]
@pytest.mark.parametrize("train", TRAIN)
def test_cifar10_dataset(train):
    dataset = kim.data.CIFAR10Dataset("data/cifar-10-batches-py", train=train)
    if train:
        assert len(dataset) == 50000
    else:
        assert len(dataset) == 10000
    example = dataset[np.random.randint(len(dataset))]
    assert(isinstance(example, tuple))
    X, y = example
    assert isinstance(X, np.ndarray)
    assert X.shape == (3, 32, 32)


BATCH_SIZES = [1, 15]
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("train", TRAIN)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_cifar10_loader(batch_size, train, device):
    cifar10_train_dataset = kim.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = kim.data.DataLoader(cifar10_train_dataset, batch_size)
    for (X, y) in train_loader:
        break
    assert isinstance(X.cached_data, nd.NDArray)
    assert isinstance(X, kim.Tensor)
    assert isinstance(y, kim.Tensor)
    assert X.dtype == 'float32'


BPTT = [3, 32]
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("bptt", BPTT)
@pytest.mark.parametrize("train", TRAIN)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_ptb_dataset(batch_size, bptt, train, device):
    # TODO update with more tests?
    corpus = kim.data.Corpus("data/ptb")
    if train:
        data = kim.data.batchify(corpus.train, batch_size, device=device, dtype="float32")
    else:
        data = kim.data.batchify(corpus.test, batch_size, device=device, dtype="float32")
    X, y = kim.data.get_batch(data, np.random.randint(len(data)), bptt, device=device)
    assert X.shape == (bptt, batch_size)
    assert y.shape == (bptt * batch_size,)
    assert isinstance(X, kim.Tensor)
    assert X.dtype == 'float32'
    assert X.device == device
    assert isinstance(X.cached_data, nd.NDArray)
    ntokens = len(corpus.dictionary)
    assert ntokens == 10000

def test_dict():
    a = kim.data.Dictionary()
    chars = "wordswordswordswordswordswordswords"
    for i in range(len(chars) - 2):
        w = chars[0:i + 2]
        print(">>>", w)
        assert a.find(w) is None
        assert a.add_word(w) == i
        assert a.add_word(w) == i
        assert a.add_word(w) == i
        assert a.find(w) == i
        assert a.lookup(i) == w

    for i in range(len(chars) - 2):
        w = chars[0:i + 2]
        print(">>>", w)
        assert a.find(w) is i
        assert a.add_word(w) == i
        assert a.add_word(w) == i
        assert a.add_word(w) == i
        assert a.find(w) == i
        assert a.lookup(i) == w


### MUGRADE ###

TEST_BATCH_SIZES = [3, 5]
TEST_BPTT = [6, 10]

def mugrade_submit(x):
    if isinstance(x, np.ndarray):
        x = x.flatten()[:128]
        #print(x)
        mugrade.submit(x)
    else:
        #print(x)
        mugrade.submit(x)


def submit_cifar10():
    if not kim.cuda().enabled():
        print('You need a GPU to run some of these tests.')
    devices = [kim.cpu(), kim.cuda()]
    for train in TRAIN:
        dataset = kim.data.CIFAR10Dataset("data/cifar-10-batches-py", train=train)
        mugrade_submit(len(dataset))
        for (device, batch_size) in itertools.product(devices, TEST_BATCH_SIZES):
            loader = kim.data.DataLoader(dataset, batch_size)
            for (X, y) in loader:
                break
            mugrade_submit(X.numpy()[0, :, :, :])
            mugrade_submit(y.numpy()[0])


def submit_ptb():
    # devices = [kim.cpu(), kim.cuda()] if kim.cuda().enabled() else [kim.cpu()]
    devices = [kim.cpu(), kim.cuda()]

    corpus = kim.data.Corpus("data/ptb")
    mugrade_submit(np.array(len(corpus.dictionary)))
    for train in TRAIN:
        for (device, batch_size, bptt) in itertools.product(devices, TEST_BATCH_SIZES, TEST_BPTT):
            if train:
                data = kim.data.batchify(corpus.train, batch_size, device=device, dtype="float32")
            else:
                data = kim.data.batchify(corpus.test, batch_size, device=device, dtype="float32")
            X, y = kim.data.get_batch(data, np.random.randint(len(data)), bptt)
            mugrade_submit(np.array(len(data)))
            mugrade_submit(X.numpy()[0, :])
            mugrade_submit(y.numpy()[0])


if __name__ == "__main__":
    submit_cifar10()
    submit_ptb()