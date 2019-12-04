import torch
import torch.nn as nn
from collections import defaultdict

class NeiLoss(nn.Module):
    def __init__(self, feat_dim=768):
        super(NeiLoss, self).__init__()
        self.feat_dim = feat_dim
        self.simiFunc = nn.Softmax(dim=0)

    def __calDis(self, x, y):#246s
        # x, y = F.normalize(qFeature), F.normalize(gFeature)
        # x, y = qFeature, gFeature
        m, n = x.shape[0], y.shape[0]
        disMat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        disMat.addmm_(1, -2, x, y.t())
        return disMat

    def forward(self, x, labels):
        bSize = x.shape[0]
        labelMap = defaultdict(list)
        distmat = self.__calDis(x,x)
        # per-ID features
        labVal = [int(val) for val in labels.cpu()]
        for pid in set(labVal):
            labelMap[pid].append(labels==pid)
        # cal loss
        loss = 0
        for keyNum in labelMap.keys():
            mask = labelMap[keyNum]
            curProb = distmat[labels==keyNum][0]
            loss += -torch.log(self.simiFunc(curProb)[mask].sum()).sum()
        return loss/len(labelMap.keys())
