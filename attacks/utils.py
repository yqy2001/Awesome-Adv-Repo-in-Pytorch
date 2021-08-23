"""
@Author:        禹棋赢
@StartTime:     2021/8/3 20:45
@Filename:      utils.py
"""
import torch


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
