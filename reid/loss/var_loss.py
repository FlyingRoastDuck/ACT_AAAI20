import torch
import torch.nn as nn
from collections import defaultdict


class VarLoss(nn.Module):
    def __init__(self, feat_dim=768):
        super(VarLoss, self).__init__()
        self.feat_dim = feat_dim
        self.simiFunc = nn.Softmax(dim=0)

    def __calDis(self, x, y):  # 246s
        # x, y = F.normalize(qFeature), F.normalize(gFeature)
        # x, y = qFeature, gFeature
        m, n = x.shape[0], y.shape[0]
        disMat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                 torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        disMat.addmm_(1, -2, x, y.t())
        return disMat

    def forward(self, x, labels):
        labelMap = defaultdict(list)
        # per-ID features
        labVal = [int(val) for val in labels.cpu()]
        for pid in set(labVal):
            labelMap[pid].append(x[labels == pid, :])
        # cal loss
        loss = 0
        for keyNum in labelMap.keys():
            meanVec = labelMap[keyNum][0].mean(dim=0, keepdim=True)
            dist = self.__calDis(meanVec, labelMap[keyNum][0])
            import ipdb;
            ipdb.set_trace()
            loss += dist.mean()
        return loss
