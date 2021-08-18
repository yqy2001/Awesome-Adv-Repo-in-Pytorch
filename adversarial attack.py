"""
@Author:        禹棋赢
@StartTime:     2021/8/2 19:37
@Filename:      adversarial attack.py
"""
import argparse

import torch
import torch.nn as nn
from tqdm import tqdm
from easydict import EasyDict

from attacks.FGM import fast_gradient_method
from attacks.PGD import projected_gradient_descent
from datasets.mnist import load_mnist
from networks.CNN import CNN
from networks.CNN_reverse import CNN_reverse


def train_attack():
    data = load_mnist()  # Load train and test data

    if args.model == "CNN":
        model = CNN(in_channels=1)  # mnist数据集通道数为1
    elif args.model == "CNN_R":
        model = CNN_reverse(in_channels=1)  # mnist数据集通道数为1
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = model.cuda()

    model.train()  # train mode
    for epoch in range(1, args.epoch+1):
        train_loss = 0.0
        for x, y in tqdm(data.train):
            x, y = x.to(device), y.to(device)
            
            output = model(x)
            
            optimizer.zero_grad()
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print("epoch: {}/{}, train loss: {:.3f}".format(epoch, args.epoch, train_loss))

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
    parser.add_argument("--model", type=str, default="CNN_R", help="which attack method to use")
    args = parser.parse_args()

    train_attack()
