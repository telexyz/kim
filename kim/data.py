import os
import gzip
import numpy as np
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        if (flip_img): return np.flip(img, axis=1)
        return img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, 
            high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        start_x = shift_x + self.padding
        start_y = shift_y + self.padding
        pad_img = np.pad(img, 
            ((self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        return pad_img[
            start_x : start_x + img.shape[0],
            start_y : start_y + img.shape[1],
            :, # same as before padding
        ]
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, img):
        if self.transforms:
            x = img.reshape((28, 28, 1))
            for tform in self.transforms: x = tform(x) # apply the transforms
            img = x.reshape(img.shape)
        return img


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        device=None,
    ):
        self.device = device
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        if self.shuffle:
            a = np.arange(len(self.dataset))
            np.random.shuffle(a)
            self.ordering = np.array_split(\
                a, range(self.batch_size, len(self.dataset), self.batch_size))
        self.n = 0
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.n >= len(self.ordering): raise StopIteration
        order = self.ordering[self.n]
        self.n += 1

        batch_xy = [self.dataset[i] for i in order]
        batch_x = Tensor([xy[0] for xy in batch_xy], device=self.device)

        if len(batch_xy[0]) == 1:
            return (batch_x,)
        else:
            batch_y = Tensor([xy[1] for xy in batch_xy], device=self.device)
            return (batch_x, batch_y)
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.transforms = transforms
        self.image_filename = image_filename
        self.label_filename = label_filename

        with gzip.open(self.image_filename) as f:
            pixels = np.frombuffer(f.read(), 'B', offset=16) # skip first 16-bytes
            self.images = pixels.reshape(-1, 28*28).astype('float32') / 255

        with gzip.open(self.label_filename) as f:
            self.labels = np.frombuffer(f.read(), 'B', offset=8) # skip 8-bytes

        self.cached_len = len(self.images)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        if isinstance(index, slice):
            images = []
            labels = []
            for idx in range(index.start, index.stop):
                images.append(self.apply_transforms(self.images[idx]))
                labels.append(self.labels[idx])
            return (np.array(images), np.array(labels))
        else:
            image = self.apply_transforms(self.images[index])
            return (image, self.labels[index])

    def __len__(self) -> int:
        return self.cached_len


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])


#########
# cifar #
#########

import pickle
'''
data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue.
labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
'''

# https://www.cs.toronto.edu/~kriz/cifar.html
class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str = "data/cifar-10-batches-py",
        train: bool = True,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION

        if train is True:
            files = ["data_batch_1", "data_batch_2",
                "data_batch_3", "data_batch_4", "data_batch_5"]
        else:
            files = ["test_batch"]

        self.train = train
        self.p = p
        self.transforms = transforms
        self.images = []
        self.labels = []
        self.length = 0

        for file in files:
            datadict = pickle.load(open(f"{base_folder}/{file}","rb"),encoding='bytes')
            imgs = datadict[b'data'].reshape((-1,3,32,32)).astype('float32')/255
            self.labels.extend(datadict[b'labels'])
            self.images.extend(imgs)
            self.length += len(imgs)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        image = self.apply_transforms(self.images[index])
        return (image, self.labels[index])
        ### END YOUR SOLUTION

    def __len__(self) -> int: return self.length


class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def find(self, word):
        try: idx = self.word2idx[word]
        except KeyError: idx = None
        return idx

    def lookup(self, idx):
        return self.idx2word[idx]

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        ### BEGIN YOUR SOLUTION
        try: n = self.word2idx[word]
        except KeyError:
            n = len(self.idx2word)
            self.word2idx[word] = n
            self.idx2word.append(word)
        return n
        ### END YOUR SOLUTION

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        ### BEGIN YOUR SOLUTION
        return len(self.idx2word)
        ### END YOUR SOLUTION



class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    def __init__(self, base_dir="./data/ptb", max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        ### BEGIN YOUR SOLUTION
        txt = open(path).read()
        lines = txt.split("\n")
        if max_lines is None: max_lines = len(lines) - 1 # remove last blank line
        assert max_lines < len(lines)

        eos_id = self.dictionary.add_word('<eos>')
        ids = []
        for i in range(max_lines):
            words = lines[i].split(" ")[1:-1] # remove first and last blank words
            for word in words:
                if len(word) > 0: ids.append(self.dictionary.add_word(word))
            ids.append(eos_id)
        return ids
        ### END YOUR SOLUTION

    def lookup(self, idx):
        return self.dictionary.idx2word[idx]


def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.

    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.

    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    ### BEGIN YOUR SOLUTION
    nbatch = len(data) // batch_size
    n = nbatch * batch_size
    # assert n == len(data)
    return np.array(data[0:n]).astype(dtype).reshape((batch_size, nbatch)).transpose()
    ### END YOUR SOLUTION


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘

    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.

    Inputs:
        * batches - numpy array returned from batchify function
        * i - index
        * bptt - Sequence length

    Returns:
        * data - Tensor of shape (bptt, bs) with cached data as NDArray
        * target - Tensor of shape (bptt * bs,) with cached data as NDArray
    """
    ### BEGIN YOUR SOLUTION
    n = batches.shape[0]
    assert i < n
    assert bptt > 0 and bptt < n - i
    
    data = batches[i : i+bptt, : ]
    i += 1
    target = batches[i : i+bptt, : ].flatten()

    return Tensor(data, device=device, dtype=dtype), Tensor(target, device=device, dtype=dtype)
    ### END YOUR SOLUTION
