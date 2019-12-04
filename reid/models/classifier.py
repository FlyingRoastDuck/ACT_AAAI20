from __future__ import absolute_import

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.autograd import Variable
import torchvision
# from torch_deform_conv.layers import ConvOffset2D
from reid.utils.serialization import load_checkpoint, save_checkpoint


class classifier(nn.Module):
    def __init__(self, inDim, outDim):
        super(classifier,self).__init__()
        self.cls = nn.Linear(inDim, outDim)
        # self.cls = nn.Sequential(
        #     nn.Linear(inDim, outDim),
        #     # nn.ReLU()
        # )
        init.normal(self.cls.weight, std=0.001)
        init.constant(self.cls.bias, 0)

    def forward(self, x):
        return self.cls(x)