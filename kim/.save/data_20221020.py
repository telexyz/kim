import numpy as np
import gzip
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
        ### BEGIN YOUR SOLUTION
        if flip_img:
            img = np.flip(img, 1)
        return img
        ### END YOUR SOLUTION


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
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        p = self.padding
        shift_x, shift_y = shift_x+p, shift_y+p
        res = np.pad(img, ((p, p), (p, p), (0, 0)))[shift_x:img.shape[0]+shift_x,shift_y:img.shape[1]+shift_y,:]
        # print(res)
        return res
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

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


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
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)),
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self.index = 0
        if self.shuffle:
            indices = np.arange(len(self.dataset))
            np.random.shuffle(indices)
            self.ordering = np.array_split(indices,
                                           range(self.batch_size, len(self.dataset), self.batch_size))

        if len(self.ordering) > 1:
            if len(self.ordering[-1]) < len(self.ordering[-2]):
                del self.ordering[-1]
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.index >= len(self.ordering):
            raise StopIteration

        data = []

        if len(self.dataset[0]) == 2:
            for el in self.ordering[self.index]:
                if isinstance(self.dataset[el], tuple):
                    for inst in zip([self.dataset[el][0]], [self.dataset[el][1]]):
                        data.append(inst)
                else:
                    for inst in zip(self.dataset[el][0], self.dataset[el][1]):
                        data.append(inst)

                imgs = []
                labels = []
                for inst in data:
                    imgs.append(inst[0])
                    labels.append(inst[1])

            data = (np.array(imgs), np.array(labels))
            data = [Tensor(d) for d in data]
        else:
            for el in self.ordering[self.index]:
                data.append(self.dataset[el])

            data = np.concatenate(data, axis=0)
            data = [Tensor(data)]

        self.index += 1

        return data
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        with gzip.open(image_filename, 'r') as f:
            magic_number = int.from_bytes(f.read(4), 'big')
            image_count = int.from_bytes(f.read(4), 'big')
            row_count = int.from_bytes(f.read(4), 'big')
            column_count = int.from_bytes(f.read(4), 'big')

            image_data = f.read()
            images = np.frombuffer(image_data, dtype=np.uint8)\
                        .reshape((image_count, row_count * column_count))
            images = images.astype(np.float32)
            images = images / 255.0

        with gzip.open(label_filename, 'r') as f:
            magic_number = int.from_bytes(f.read(4), 'big')
            label_count = int.from_bytes(f.read(4), 'big')
            label_data = f.read()
            labels = np.frombuffer(label_data, dtype=np.uint8)

        self.images = images
        self.labels = labels
        self.transforms = transforms if transforms else []
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        res = []
        imgs = []
        labels = []
        if isinstance(index, slice):
            for ind in range(index.start, index.stop):
                img = self.images[ind].reshape(28, 28, 1)
                img = self.apply_transforms(img).flatten()
                imgs.append(img)
                labels.append(self.labels[ind])
        else:
            img = self.images[index].reshape(28, 28, 1)
            img = self.apply_transforms(img).flatten()
            return (img, self.labels[index])

        return (np.array(imgs), np.array(labels))
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.images.shape[0]
        ### END YOUR SOLUTION

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
