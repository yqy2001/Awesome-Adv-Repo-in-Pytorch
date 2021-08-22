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

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

mu_tensor = torch.tensor(cifar10_mean).view(3,1,1)
std_tensor = torch.tensor(cifar10_std).view(3,1,1)


def load_cifar10(bsz, mode=0):
    """
    data transform follow FastAT
    Load training and test data of cifar10
    each image is [3, 32, 32]
    """

    if mode == 0:
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])
    elif mode == 1:
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])
    elif mode == 2:
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(cifar10_mean, cifar10_std),
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(cifar10_mean, cifar10_std),
        ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=_DATA_cifar10, train=True, transform=train_transforms, download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=_DATA_cifar10, train=False, transform=test_transforms, download=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bsz, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=bsz, shuffle=False, num_workers=2
    )
    return EasyDict(train=train_loader, test=test_loader)
