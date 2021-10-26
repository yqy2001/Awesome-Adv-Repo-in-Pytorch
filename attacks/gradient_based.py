"""
@Author:        禹棋赢
@StartTime:     2021/8/3 20:13
@Filename:      PGD.py
"""
import torch


def projected_gradient_descent(model_fn, loss_fn, x, y, eps=8./255, alpha=2./255, nb_iter=10, random_init=True):
    """
    If random_init is false, then this function is I-FGSM or BIM; else PGD.
    :param model_fn:
    :param loss_fn:
    :param x:
    :param y:
    :param eps:
    :param nb_iter:
    :param alpha: step size to update perturbation at each iteration
    :param random_init: whether to randomly initialize
    :return:
    """

    if random_init:  # pgd
        perturbation = torch.zeros_like(x).uniform_(-eps, eps)
        perturbation = clip_perturbation(perturbation, eps)
    else:
        perturbation = torch.zeros_like(x)

    perturbation = torch.nn.Parameter(perturbation)  # requires_grad
    x_adv = x + perturbation

    for i in range(nb_iter):
        # x_adv = x_adv.clone().detach().to(torch.float).requires_grad_(True)  # 对x求梯度
        data, _ = model_fn(x_adv)
        loss = loss_fn(data, y)

        model_fn.zero_grad()
        loss.backward()
        perturbation.data += alpha * torch.sign(perturbation.grad)
        perturbation.grad = None  # zero grad for ptb
        perturbation.data = clip_perturbation(perturbation.data, eps)
        # perturbation.data = torch.clamp(x + perturbation.data, min=0, max=1) - x  # todo not sure whether need to  do this

        x_adv = x_adv + perturbation

    return x_adv.detach()  # dont need grad then


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


def clip_perturbation(pb, eps, norm='inf'):
    """
    clip the perturbation according to different norms
    :param pb: perturbation to clip
    :param eps: the constraints bound
    :param norm: norm used to measure constraints, possible values: "inf", 1, 2
    :return: the cliped perturbation
    """
    if norm == "inf":
        if torch.is_tensor(eps):
            for i in range(eps.shape[0]):
                pb[:, i, :, :] = torch.clip(pb[:, i, :, :], -eps[i], eps[i])
            return pb
        else:
            return torch.clamp(pb, -eps, eps)
