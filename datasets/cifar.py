"""
@Author:        禹棋赢
@StartTime:     2021/8/4 18:51
@Filename:      cifar.py
"""
import torch
import torchvision
import torchvision.transforms as transforms
from easydict import EasyDict

_DATA_cifar10 = "datasets/cifar10/"
_DATA_cifar100 = "datasets/cifar100/"


def load_cifar10():
    """
    Load training and test data of cifar10
    each image is [3, 32, 32]
    """
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])
    train_dataset = torchvision.datasets.CIFAR10(
        root=_DATA_cifar10, train=True, transform=train_transforms, download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=_DATA_cifar10, train=False, transform=test_transforms, download=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=2
    )
    return EasyDict(train=train_loader, test=test_loader)
