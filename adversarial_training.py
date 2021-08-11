"""
@Author:        禹棋赢
@StartTime:     2021/8/4 18:14
@Filename:      adversarial_training.py
"""
import numpy as np
import random

import time

import shutil

import os

import argparse
import torch
import torchvision
from easydict import EasyDict
from tqdm import tqdm

from attacks.PGD import projected_gradient_descent
from attacks.utils import clip_perturbation
from datasets.cifar import load_cifar10
from networks.resnet import ResNet50


class AT(object):

    def __init__(self, dataset, model, args):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = model
        self.model.to(self.device)
        if self.device == 'cuda':
            self.model = torch.nn.DataParallel(self.model)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9, weight_decay=2e-4)

        self.args = args
        self.dataset = dataset

        self.start_epoch = 1
        self.epoch = 0
        self.best_acc = 0.0

        self.train_time = 0
        self.freeat_perturbation = 0  # freeat's perturbation is not reset between epochs and minibatches

        if args.resume:
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/ckpt.pth')
            self.model.load_state_dict(checkpoint['model'])
            self.best_acc = checkpoint['adv_acc']
            self.start_epoch = checkpoint['epoch']

    def process(self):
        if self.args.adv_train_method == "FreeAT":
            epochs = int(self.args.epochs / self.args.K)
        else:
            epochs = self.args.epochs
        for epoch in range(self.start_epoch, epochs + 1):
            print("\nEpoch: %d/%d" % (epoch, self.args.epochs))
            self.epoch = epoch
            # adjust_learning_rate(self.optimizer, self.epoch)
            start = time.time()
            if self.args.adv_train_method == "natural":
                self.train_natural()
            elif self.args.adv_train_method == "FreeAT":
                self.train_FreeAT()
            elif self.args.adv_train_method == "VanillaPGD":
                self.train_VanillaPGD()
            self.train_time += (time.time()-start)
            self.test()
            print(self.args.adv_train_method + (" training time is {:2f}".format(self.train_time)))

    def train_natural(self):
        self.model.train()
        train_loss = 0.0
        for x, y in self.dataset.train:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model(x), y)
            loss.backward()
            self.optimizer.step()  # update NN params

            train_loss += loss.item()
        print("train loss is {:2f}".format(train_loss))

    def train_VanillaPGD(self):
        """
        every time update the images, update the NN params by the way

        Experiment Settings:  (in original PGD AT paper)
        eps=8

        :return:
        """
        self.model.train()
        train_loss = 0.0
        for x, y in self.dataset.train:
            x, y = x.to(self.device), y.to(self.device)
            perturbation = torch.zeros_like(x).uniform_(-self.args.eps, self.args.eps)  # PGD's random initialization
            # update images
            for i in range(self.args.K):
                self.optimizer.zero_grad()
                x_adv = (x + perturbation).detach().requires_grad_(True)
                loss = self.loss_fn(self.model(x_adv), y)
                loss.backward()
                perturbation = clip_perturbation(perturbation + self.args.step_size * torch.sign(x_adv.grad), self.args.eps)

                train_loss += loss.item()

            self.optimizer.zero_grad()
            x_adv = (x + perturbation).detach().requires_grad_(True)
            loss = self.loss_fn(self.model(x_adv), y)
            loss.backward()
            self.optimizer.step()  # update NN
        print("train loss is {:2f}".format(train_loss))

    def train_FreeAT(self):
        """
        every time update the images, update the NN params by the way
        perturbations are not reset between epochs and minibatches
        AE update step size is eps not alpha
        :return:
        """
        self.model.train()
        train_loss = 0.0
        for x, y in self.dataset.train:
            x, y = x.to(self.device), y.to(self.device)
            # todo maybe different perturbations for different minibatches
            for i in range(self.args.K):
                self.optimizer.zero_grad()
                x_adv = (x + self.freeat_perturbation).detach().requires_grad_(True)
                loss = self.loss_fn(self.model(x_adv), y)
                loss.backward()
                self.optimizer.step()  # update NN params
                # per = per + eps * sign(grad)
                self.freeat_perturbation = clip_perturbation(self.freeat_perturbation + self.args.eps * torch.sign(x_adv.grad), self.args.eps)

                train_loss += loss.item()

        print("train loss is {:2f}".format(train_loss))

    def test(self):
        self.model.eval()
        metrics = EasyDict(total=0, correct=0, correct_adv=0)
        for x, y in self.dataset.test:
            x, y = x.to(self.device), y.to(self.device)
            _, pred = self.model(x).max(1)
            metrics.total += y.shape[0]
            metrics.correct += torch.eq(pred, y).sum().item()

            # construct AE
            perturbation = torch.zeros_like(x).uniform_(-self.args.eps, self.args.eps)  # PGD's random initialization
            for i in range(self.args.K):
                self.optimizer.zero_grad()
                x_adv = (x + perturbation).detach().requires_grad_(True)
                loss = self.loss_fn(self.model(x_adv), y)
                loss.backward()
                perturbation = clip_perturbation(perturbation + self.args.lr * torch.sign(x_adv.grad), self.args.eps)
            x_adv = (x + perturbation).detach().requires_grad_(True)
            _, pred = self.model(x_adv).max(1)
            metrics.correct_adv += torch.eq(pred, y).sum().item()

        acc = 100.0 * metrics.correct / metrics.total
        adv_acc = 100.0 * metrics.correct_adv / metrics.total
        print("test acc is {:2f}, Adversarial Examples acc is {:2f}".format(acc, adv_acc))
        if self.args.adv_train_method == "natural":  # natural training, use acc to measure model performance
            if acc > self.best_acc:
                self.best_acc = acc
                print("This epoch's acc is better, Saving...")
                if not os.path.isdir("checkpoint"):
                    os.mkdir("checkpoint")
                torch.save({
                    'model': self.model.state_dict(),
                    'adv_acc': acc,
                    'epoch': self.epoch,
                }, "./checkpoint/ckpt.pth")
        else:  # adversarial training, use adv_acc to measure
            if adv_acc > self.best_acc:
                self.best_acc = adv_acc
                print("This epoch's AE acc is better, Saving...")
                if not os.path.isdir("checkpoint"):
                    os.mkdir("checkpoint")
                torch.save({
                    'model': self.model.state_dict(),
                    'adv_acc': adv_acc,
                    'epoch': self.epoch,
                }, "./checkpoint/ckpt.pth")


# def adjust_learning_rate(optimizer, epoch):
#     if epoch <= 60:
#         lr = 0.1
#     elif 60 <= epoch < 120:
#         lr = 0.01
#     elif epoch >= 120:
#         lr = 0.001
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=5252520, type=int)

    parser.add_argument('--eps', default=8.0/255, type=float)
    parser.add_argument('--step_size', default=2.0/255, type=float, help="step size to update AE")
    parser.add_argument("--epochs", type=int, default=200, help='iter nums to train NN')
    parser.add_argument("--lr", type=float, default=0.08, help='learning rate')
    parser.add_argument("--adv_train_method", "-atm", type=str, default="FreeAT", choices=["Natural", "VanillaPGD", "FreeAT"], help="adversarial training type")
    parser.add_argument("--resume", '-r', action='store_true', help="resume training from checkpoint")
    parser.add_argument('--K', default=7, type=int)
    args = parser.parse_args()

    # fix seed
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    cifar10 = load_cifar10()
    model = ResNet50()
    at = AT(cifar10, model, args)
    at.process()


if __name__ == '__main__':
    run()
