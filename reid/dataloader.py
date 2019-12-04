from torch.autograd import Variable
import numpy as np
import torch


class GraphLoader(object):
    def __init__(self, trainList, labels, model, loss=None):
        # self.hw = [384,128]
        self.graphs = trainList
        self.ID = labels
        self.model = model
        self.loss = loss

    def __getitem__(self, idx):
        curGraph = self.graphs[idx]
        # featSize = curGraph.size(0)
        # useFeat = curGraph[np.random.choice(featSize, size=(int(0.8*featSize),), replace=False), :]
        gEmb, scores = self.model(curGraph)
        loss = self.loss(scores, torch.LongTensor([self.ID[idx]]).cuda()) if self.loss is not None else 0
        return gEmb.squeeze(), loss  # return embedding and loss

    def __len__(self):
        return len(self.graphs)
