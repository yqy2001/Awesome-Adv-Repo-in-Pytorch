"""
@Author:        禹棋赢
@StartTime:     2021/8/2 19:18
@Filename:      mnist.py

from cleverhans
"""
import array
import gzip
import os
from os import path
import struct
from urllib.request import urlretrieve

import torch
import torchvision
from easydict import EasyDict
import numpy as np

_DATA = "datasets/mnist/"


def load_mnist():
    """Get MNIST training and test dataloader."""
    train_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    test_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )

    # Load MNIST dataset
    train_dataset = MNISTDataset(root=_DATA, transform=train_transforms)
    test_dataset = MNISTDataset(
        root=_DATA, train=False, transform=test_transforms
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=2
    )
    return EasyDict(train=train_loader, test=test_loader)


class MNISTDataset(torch.utils.data.Dataset):

    def __init__(self, root=_DATA, train=True, transform=None):

        train_images, train_labels, test_images, test_labels = self.get_mnist_data(root)

        if train:
            self.images = train_images
            self.labels = torch.from_numpy(train_labels).long()
        else:
            self.images = test_images
            self.labels = torch.from_numpy(test_labels).long()

        self.transform = transform

    def __getitem__(self, index):
        x = self.images[index]
        if self.transform:
            x = self.transform(x)
        return x, self.labels[index]

    def __len__(self):
        return len(self.images)

    def get_mnist_data(self, root):
        """
        download the mnist dataset from the official website
        :param root: the folder to restore the dataset
        """
        base_url = "http://yann.lecun.com/exdb/mnist/"

        def parse_labels(filename):
            with gzip.open(filename, "rb") as fh:
                _ = struct.unpack(">II", fh.read(8))
                return np.array(array.array("B", fh.read()), dtype=np.uint8)

        def parse_images(filename):
            with gzip.open(filename, "rb") as fh:
                _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
                return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(
                    num_data, rows, cols
                )

        def download(url, filename):
            """Download a url to a file in the JAX data temp directory."""
            if not path.exists(root):
                os.makedirs(root)
            out_file = path.join(root, filename)
            if not path.isfile(out_file):
                urlretrieve(url, out_file)
                print("downloaded {} to {}".format(url, root))

        for filename in [
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
        ]:
            download(base_url + filename, filename)

        train_images = parse_images(path.join(root, "train-images-idx3-ubyte.gz"))
        train_labels = parse_labels(path.join(root, "train-labels-idx1-ubyte.gz"))
        test_images = parse_images(path.join(root, "t10k-images-idx3-ubyte.gz"))
        test_labels = parse_labels(path.join(root, "t10k-labels-idx1-ubyte.gz"))

        return train_images, train_labels, test_images, test_labels
