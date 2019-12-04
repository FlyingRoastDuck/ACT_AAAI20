#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import cdist
import torch
import tqdm
import torch.nn.functional as F


def pairwiseDis(qFeature, gFeature):  # 246s
    x, y = F.normalize(qFeature), F.normalize(gFeature)
    # x, y = qFeature, gFeature
    m, n = x.shape[0], y.shape[0]
    disMat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
             torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    disMat.addmm_(1, -2, x, y.t())
    print('-----* Distance Matrix has been computed*-----')
    return disMat.clamp_(min=1e-5)


def re_ranking(input_feature_source, input_feature, k1=20, k2=6, lambda_value=0.1):
    all_num = input_feature.shape[0]
    # feat = input_feature.astype(np.float16)
    feat = torch.from_numpy(input_feature)  # target
    del input_feature

    if lambda_value != 0:
        print('Computing source distance...')
        srcFeat, tarFeat = input_feature_source, feat
        # all_num_source  = input_feature_source.shape[0]
        # sour_tar_dist = np.power(cdist(input_feature, input_feature_source), 2).astype(np.float32) #608s
        sour_tar_dist = pairwiseDis(srcFeat, tarFeat).t().numpy()
        sour_tar_dist = 1 - np.exp(-sour_tar_dist)  # tar-src
        source_dist_vec = np.min(sour_tar_dist, axis=1)
        source_dist_vec = source_dist_vec / (np.max(source_dist_vec) + 1e-3)  # for trget
        source_dist = np.zeros([all_num, all_num])  # tar size
        for i in range(all_num):
            source_dist[i, :] = source_dist_vec + source_dist_vec[i]
        del sour_tar_dist
        del source_dist_vec

    print('Computing original distance...')
    original_dist = pairwiseDis(feat, feat).cpu().numpy()
    print('done...')
    # original_dist = np.power(original_dist,2).astype(np.float16)
    del feat
    # original_dist = np.concatenate(dist,axis=0)
    gallery_num = original_dist.shape[0]  # gallery_num=all_num
    original_dist = np.transpose(original_dist / (np.max(original_dist, axis=0)))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)  ## default axis=-1.  

    print('Starting re_ranking...')
    for i in tqdm.tqdm(range(all_num)):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,
                                :k1 + 1]  ## k1+1 because self always ranks first. forward_k_neigh_index.shape=[k1+1].  forward_k_neigh_index[0] == i.
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,
                                 :k1 + 1]  ##backward.shape = [k1+1, k1+1]. For each ele in forward_k_neigh_index, find its rank k1 neighbors
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]  ## get R(p,k) in the paper
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  ## element-wise unique
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    # original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in tqdm.tqdm(range(all_num)):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in tqdm.tqdm(range(gallery_num)):
        invIndex.append(np.where(V[:, i] != 0)[0])  # len(invIndex)=all_num

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in tqdm.tqdm(range(all_num)):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0

    if lambda_value == 0:
        return jaccard_dist
    else:
        final_dist = jaccard_dist * (1 - lambda_value) + source_dist * lambda_value
        return final_dist
