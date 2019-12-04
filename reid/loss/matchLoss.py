import torch
import torch.nn.functional as F

def pairwiseDis(x, y):#246s
    # x, y = F.normalize(qFeature), F.normalize(gFeature)
    # x, y = qFeature, gFeature
    m, n = x.shape[0], y.shape[0]
    disMat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
             torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    disMat.addmm_(1, -2, x, y.t())
    return disMat


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = source.shape[0]+target.shape[0] 
    L2_distance = pairwiseDis(source, target)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)/len(kernel_val)

def lossMMD(srcFeat, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    srcBatch = srcFeat.shape[0]
    newFeat = torch.cat([srcFeat,target])
    kernels = guassian_kernel(newFeat, newFeat, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:srcBatch, :srcBatch]
    YY = kernels[srcBatch:, srcBatch:]
    XY = kernels[:srcBatch, srcBatch:]
    YX = kernels[srcBatch:, :srcBatch]
    return XX.mean() + YY.mean() - XY.mean() -YX.mean()


# if __name__ == "__main__":
#     import numpy as np
#     srcFeat, tarFeat = np.load('E://gcn//srcGFeat.npy'), np.load('E://gcn//tarGFeat.npy')
#     disMat = lossMMD(torch.from_numpy(srcFeat), torch.from_numpy(tarFeat))