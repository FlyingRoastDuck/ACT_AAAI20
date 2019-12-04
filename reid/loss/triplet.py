from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable


class TripletLoss(nn.Module):
    def __init__(self, margin=0, num_instances=0, use_semi=True, isAvg=True):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=self.margin, reduce=isAvg)
        self.K = num_instances
        self.use_semi = use_semi

    def forward(self, inputs, targets, epoch):
        n = inputs.size(0)
        P = n // self.K

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        if self.use_semi:
            for i in range(P):
                for j in range(self.K):
                    neg_examples = dist[i * self.K + j][mask[i * self.K + j] == 0]
                    for pair in range(j + 1, self.K):
                        ap = dist[i * self.K + j][i * self.K + pair]
                        dist_ap.append(ap.unsqueeze(0))
                        dist_an.append(neg_examples.min().unsqueeze(0))
        else:
            for i in range(n):
                dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
                dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec
