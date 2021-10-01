"""
@Author:        禹棋赢
@StartTime:     2021/8/2 19:15
@Filename:      FGM.py
"""
import torch


def fast_gradient_method(model_fn, loss_fn, x, y, eps, norm="inf"):
    """
    currently only complement white box attack to model_fn, and non-target attack to y, and l-inf norm constraint

    :param model_fn: model function
    :param loss_fn: loss function
    :param x: clean example
    :param y: original label corresponding to x
    :param eps: constraint
    :param norm: which norm to measure distance, choices: [1, 2, "inf"]
    :return:
    """
    # x needs to be a leaf variable, of floating point type and have requires_grad being True for
    # its grad to be computed and stored properly in a backward call
    x = x.clone().detach().to(torch.float).requires_grad_(True)
    data, _ = model_fn(x)
    loss = loss_fn(data, y)
    loss.backward()  # backward and compute gradients
    adv_x = x + eps * torch.sign(x.grad)
    return adv_x
