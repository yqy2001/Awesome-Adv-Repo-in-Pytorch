import argparse
import os
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from tqdm import tqdm
from easydict import EasyDict

from attacks.gradient_based import projected_gradient_descent, fast_gradient_method
from datasets.mnist import load_mnist
from networks.Kervolution import Ker_ResNet18, Ker_ResNet34
from networks.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

import time
from functools import partial
from feature_loss import Feature_loss

from tsne import tsne

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
    loss_fn_feature = Feature_loss(temperature=0.05, lambd = args.lambd)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = model.cuda()
    
    ticks0 = time.time()
    for epoch in range(1, args.epoch+1):
        model.train()  # train mode
        train_loss = 0.0
        loss_acc = 0.0
        loss_feature = 0.0

        ticks = time.time()
        for x, y in tqdm(data.train):
            x, y = x.to(device), y.to(device)
            
            output, feature = model(x)
            
            optimizer.zero_grad()
            
            loss, loss_acc1, loss_feature1 = loss_fn_feature(output, feature, y)
            
            loss.backward()
            optimizer.step()

            loss_acc += loss_acc1.item()
            loss_feature += loss_feature1.item()
            train_loss += loss.item()

        tocks = time.time()
        print("epoch: {}/{}, train loss: {:.3f}, acc_loss: {:.3f},feature_loss:{:.3f}, train cost {:.2f} s".format(epoch, args.epoch, train_loss, loss_acc, loss_feature, tocks-ticks))
        
        model.eval()
        metrics = EasyDict(correct=0, correct_fgm=0, total=0, correct_pgd=0)
        for x, y in tqdm(data.test):
            x, y = x.to(device), y.to(device)
            # if args.atm == "fgm":
            x_fgm = fast_gradient_method(model, loss_fn, x, y, args.eps)
            x_pgd = projected_gradient_descent(model, loss_fn, x, y, args.eps, args.iter, args.lr)

            output, _ = model(x)
            _, pred = output.max(1)
            output, _ = model(x_fgm)
            _, pred_fgm = output.max(1)
            output, _ = model(x_pgd)
            _, pred_pgd = output.max(1)
            metrics.total += y.shape[0]
            metrics.correct += torch.eq(pred, y).sum().item()
            metrics.correct_fgm += torch.eq(pred_fgm, y).sum().item()
            metrics.correct_pgd += torch.eq(pred_pgd, y).sum().item()

        print("test acc on clean examples (%): {:3f}".format(metrics.correct / metrics.total * 100.0))
        print("test acc on fgm examples (%): {:3f}".format(metrics.correct_fgm / metrics.total * 100.0))
        print("test acc on pgd examples (%): {:3f}".format(metrics.correct_pgd / metrics.total * 100.0))
    
    root = 'figures/'
    train_save_dir = os.path.join(root, args.traindir)+'.jpg'
    test_save_dir = os.path.join(root, args.testdir)+'.jpg'

    temp = 1
    if (os.path.exists(train_save_dir)): print('Save-dir of train feature scatter diagram exists! Please add remarks!')
    while (os.path.exists(train_save_dir)):
        train_save_dir = os.path.join(root, args.traindir)+str(temp)+'.jpg'
        temp += 1
    
    if (os.path.exists(test_save_dir)): print('Save-dir of test feature scatter diagram exists! Please add remarks!')
    temp = 1
    while (os.path.exists(test_save_dir)):
        test_save_dir = os.path.join(root, args.testdir)+str(temp)+'.jpg'
        temp += 1
    
    print("train feature scatter diagram save dir:{}".format(train_save_dir))
    print("test feature scatter diagram save dir:{}".format(test_save_dir))

    tsne(model, device, data.train, train_save_dir)
    tsne(model, device, data.test, test_save_dir)

    tocks0 = time.time()

    print("total cost {:.2f} s".format(tocks0-ticks0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--eps", type=float, default=0.3, help="eps to constrain AE (adversarial examples)")
    parser.add_argument("--iter", type=int, default=10, help='iter nums to update AE, K in PGD-K')
    parser.add_argument("--lr", type=float, default=0.01, help='learning rate of iterative attack method')
    parser.add_argument("--atm", type=str, default="pgd", help="which attack method to use")
    parser.add_argument('--lambd', type = float, default=2, help ="ratio of Feature loss to SCE loss")
    parser.add_argument("--model", type=str, default='resnet18',choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "ker_resnet18", "ker_resnet34"],help="choose a model")
    parser.add_argument("--traindir", type=str, default='train_feature_distribute', help="save-dir of train feature scatter diagram")
    parser.add_argument("--testdir", type=str, default='test_feature_distribute', help="save-dir of train feature scatter diagram")
    args = parser.parse_args()
    print('epoch:{}  eps to constrain AE:{}  iter nums to update AE:{}  learning rate of iterative attack method:{}'.format(args.epoch, args.eps, args.iter, args.lr))
    print('atm:{}  lambd(ratio of Feature loss to SCE loss):{}  model:{}'.format(args.atm, args.lambd, args.model))
    train_attack()
