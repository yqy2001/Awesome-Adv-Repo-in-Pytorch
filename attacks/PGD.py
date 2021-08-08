"""
@Author:        禹棋赢
@StartTime:     2021/8/3 20:13
@Filename:      PGD.py
"""
import torch

from attacks.utils import clip_perturbation


def projected_gradient_descent(model_fn, loss_fn, x, y, eps, nb_iter, lr, random_init=True):
    """
    If random_init is false, then this function is I-FGSM or BIM; else PGD.
    :param model_fn:
    :param loss_fn:
    :param x:
    :param y:
    :param eps:
    :param nb_iter:
    :param lr:
    :param random_init:
    :return:
    """

    if random_init:  # pgd
        perturbation = torch.zeros_like(x).uniform_(-eps, eps)
        perturbation = clip_perturbation(perturbation, eps)
        x_adv = x + perturbation
    else:
        x_adv = x

    for i in range(nb_iter):
        x_adv = x_adv.clone().detach().to(torch.float).requires_grad_(True)  # 对x求梯度
        loss = loss_fn(model_fn(x_adv), y)
        loss.backward()
        perturbation = lr * torch.sign(x_adv.grad)
        perturbation = clip_perturbation(perturbation, eps)
        x_adv = x_adv + perturbation

    return x_adv
