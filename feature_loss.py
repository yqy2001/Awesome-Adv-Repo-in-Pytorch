import torch
import torch.nn as nn
import torch.nn.functional as F


class Feature_loss(nn.Module):

    def __init__(self, temperature, lambd):
        super(Feature_loss, self).__init__()
        self.temperature = temperature
        self.lambd = lambd
        self.pdist = nn.PairwiseDistance(p=2)
        self.SCE = nn.CrossEntropyLoss()

    def cos_sim(self, matrix_i_pos, matrix_pos, matrix_i, matrix_all):
        loss_pos = torch.exp(F.cosine_similarity(matrix_i_pos, matrix_pos, dim=1) / self.temperature).sum()
        loss_total = torch.exp(F.cosine_similarity(matrix_i, matrix_all, dim=1) / self.temperature).sum()
        loss_fea = -torch.log(loss_pos / loss_total)
        return loss_fea

    def pdist_sim(self, matrix_i_pos, matrix_pos, matrix_i, matrix_all):
        loss_pos = self.pdist(matrix_i_pos, matrix_pos).sum()
        loss_total = self.pdist(matrix_i, matrix_all).sum()
        loss_fea = (loss_pos) / (loss_total - loss_pos)
        return loss_fea

    def forward(self, data, x, y):
        batch_size = x.shape[0]
        loss_acc = self.SCE(data, y)
        loss_feature = 0.0
        loss = 0.0

        for i in range(batch_size):
            label = y[i].item()
            pos = torch.nonzero(y == label)

            matrix_all = x
            matrix_i = x[i, :].repeat(batch_size, 1)

            matrix_pos = x[pos[0], :]
            matrix_i_pos = x[i, :].repeat(matrix_pos.shape[0], 1)

            # loss_fea = self.cos_sim(matrix_i_pos, matrix_pos, matrix_i, matrix_all)

            loss_fea = self.pdist_sim(matrix_i_pos, matrix_pos, matrix_i, matrix_all)

            loss_feature += loss_fea

        # loss_feature /= batch_size

        loss = loss_acc + self.lambd * loss_feature

        return loss, loss_acc, loss_feature


class Class_loss(nn.Module):
    """
    compute the in-class variance loss and inter-class distance loss
    """

    def __init__(self, class_num, alpha=0.5):
        super(Class_loss, self).__init__()
        self.class_num = class_num
        self.class_feature = [[]] * class_num

        self.pdist = nn.PairwiseDistance(p=2)

        self.alpha = alpha

    def forward(self, features, labels, centers):
        var_loss = 0.
        for i in range(len(features)):
            center = centers[labels[i]]
            var_loss += self.pdist(center.view(1, -1), features[i])

        return var_loss

    def compute_inter_loss(self, centers):
        # given centers, compute inter-class loss
        inter_loss = 0.
        for i in range(self.class_num):
            for j in range(i + 1, self.class_num):
                inter_loss += -self.pdist(centers[i].view(1, -1), centers[j].view(1, -1))

        return inter_loss
