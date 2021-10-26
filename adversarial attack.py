import argparse

import torch
import torch.nn as nn
from tqdm import tqdm
from easydict import EasyDict

from attacks.gradient_based import projected_gradient_descent, fast_gradient_method
from datasets.mnist import load_mnist
from networks.Kervolution import Ker_ResNet18, Ker_ResNet34
from networks.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

import time
from functools import partial

def train_attack():
    data = load_mnist()  # Load train and test data
    if args.model=='resnet18':
        model = ResNet18()
    elif args.model=='resnet34':
        model = ResNet34()
    elif args.model=='resnet50':
        model = ResNet50()
    elif args.model=='resnet101':
        model = ResNet101()
    elif args.model=='resnet152':
        model = ResNet152()
    elif args.model=='ker_resnet18':
        model = Ker_ResNet18()
    elif args.model=='ker_resnet34':
        model = Ker_ResNet34()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = model.cuda()

    model.train()  # train mode
    ticks0 = time.time()
    for epoch in range(1, args.epoch+1):
        train_loss = 0.0
        ticks = time.time()
        for x, y in tqdm(data.train):
            x, y = x.to(device), y.to(device)
            
            output = model(x)
            
            optimizer.zero_grad()
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        tocks = time.time()
        print("epoch: {}/{}, train loss: {:.3f}, train cost {:.2f} s".format(epoch, args.epoch, train_loss, tocks-ticks))
    tocks0 = time.time()
    print("total cost {:.2f} s".format(tocks0-ticks0))
    
    model.eval()
    metrics = EasyDict(correct=0, correct_fgm=0, total=0, correct_pgd=0)
    for x, y in data.test:
        x, y = x.to(device), y.to(device)
        # if args.atm == "fgm":
        x_fgm = fast_gradient_method(model, loss_fn, x, y, args.eps)
        x_pgd = projected_gradient_descent(model, loss_fn, x, y, args.eps, args.iter, args.lr)
        _, pred = model(x).max(1)
        _, pred_fgm = model(x_fgm).max(1)
        _, pred_pgd = model(x_pgd).max(1)
        metrics.total += y.shape[0]
        metrics.correct += torch.eq(pred, y).sum().item()
        metrics.correct_fgm += torch.eq(pred_fgm, y).sum().item()
        metrics.correct_pgd += torch.eq(pred_pgd, y).sum().item()

    print("test acc on clean examples (%): {:3f}".format(metrics.correct / metrics.total * 100.0))
    print("test acc on fgm examples (%): {:3f}".format(metrics.correct_fgm / metrics.total * 100.0))
    print("test acc on pgd examples (%): {:3f}".format(metrics.correct_pgd / metrics.total * 100.0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--eps", type=float, default=0.3, help="eps to constrain AE (adversarial examples)")
    parser.add_argument("--iter", type=int, default=40, help='iter nums to update AE, K in PGD-K')
    parser.add_argument("--lr", type=float, default=0.01, help='learning rate of iterative attack method')
    parser.add_argument("--atm", type=str, default="pgd", help="which attack method to use")
    parser.add_argument("--model", type=str, default='resnet18',choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "ker_resnet18", "ker_resnet34"],help="choose a model")
    args = parser.parse_args()

    train_attack()
