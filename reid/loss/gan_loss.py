import torch
import torch.nn as nn


class GanLoss(nn.Module):
    def __init__(self, modelD, modelG, inDim=2048, outDim=2, lossD=nn.CrossEntropyLoss(), lossG=nn.CrossEntropyLoss()):
        super(GanLoss, self).__init__()
        self.inDim = inDim
        self.outDim = outDim
        self.modelD = modelD
        self.modelG = modelG
        self.lossG = lossG
        self.lossD = lossD

    def forward(self, x, labels, domainLab):
        dScore, gScore = self.modelD(x), self.modelG(x)
        lossDomain = self.lossD(dScore, domainLab)
        lossCls = self.lossG(gScore, labels)
        return lossDomain, lossCls
