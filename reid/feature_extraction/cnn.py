from __future__ import absolute_import
from collections import OrderedDict

from ..utils import to_torch
from torch.autograd import Variable


def extract_cnn_feature(model, inputs):
    model.eval()
    inputs = Variable(to_torch(inputs))

    outputs = model(inputs)[0]
    outputs = outputs.data.cpu()
    return outputs  # pool5

    # Register forward hook for each module
    outputs = OrderedDict()
    handles = []
    for m in modules:
        outputs[id(m)] = None

        def func(m, i, o): outputs[id(m)] = o.data.cpu()

        handles.append(m.register_forward_hook(func))
    model(inputs)
    for h in handles:
        h.remove()
    return list(outputs.values())
