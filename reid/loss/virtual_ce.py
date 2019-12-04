from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from scipy.stats import norm

import numpy as np


class VirtualCE(nn.Module):
    def __init__(self, beta=0.1):
        super(VirtualCE, self).__init__()
        self.beta = beta

    def forward(self, inputs, targets):
        # norm first
        n = inputs.shape[0]
        inputs = F.normalize(inputs, p=2)
        allPids = targets.cpu().numpy().tolist()
        # All Centers
        centerHash = {
            pid: F.normalize(inputs[targets == pid, :].mean(dim=0, keepdim=True), p=2).detach() for pid in set(allPids)
        }
        allCenters = torch.autograd.Variable(torch.cat(list(centerHash.values()))).cuda()
        centerPID = torch.from_numpy(np.asarray(list(centerHash.keys())))
        # sampler vs center
        samplerCenter = torch.autograd.Variable(torch.cat([allCenters[centerPID == pid, :] for pid in allPids])).cuda()
        # inputs--(128*1024), allCenters--(32*1024)
        vce = torch.diag(torch.exp(samplerCenter.mm(inputs.t()) / self.beta))  # 1*128
        centerScore = torch.exp(allCenters.mm(inputs.t()) / self.beta).sum(dim=0)  # 32(center number)*128->1*128
        return -torch.log(vce.div(centerScore)).mean()


class VirtualKCE(nn.Module):
    def __init__(self, beta=0.1):
        super(VirtualKCE, self).__init__()
        self.beta = beta

    def forward(self, inputs, targets):
        # norm first
        n = inputs.shape[0]
        inputs = F.normalize(inputs, p=2)
        allPids = targets.cpu().numpy().tolist()
        # All Centers
        centerHash = {
            pid: F.normalize(inputs[targets == pid, :].mean(dim=0, keepdim=True), p=2).detach() for pid in set(allPids)
        }
        allCenters = torch.autograd.Variable(torch.cat(list(centerHash.values()))).cuda()
        centerPID = torch.from_numpy(np.asarray(list(centerHash.keys())))
        samplerCenter = torch.autograd.Variable(torch.cat([allCenters[centerPID == pid, :] for pid in allPids])).cuda()
        # inputs--(128*1024), allCenters--(32*1024)
        vce = torch.diag(torch.exp(samplerCenter.mm(inputs.t()) / self.beta))  # 1*128
        centerScore = torch.exp(allCenters.mm(inputs.t()) / self.beta).sum(dim=0)  # 32*128->1*128
        kNegScore = torch.diag(inputs.mm(inputs.t()))
        return -torch.log(vce.div(kNegScore + centerScore)).mean()
