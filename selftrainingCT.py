#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import
import argparse
import time
import os.path as osp
import os
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.backends import cudnn
from torch.utils.data import DataLoader
from reid import datasets
from reid import models
from reid.dist_metric import DistanceMetric
from reid.loss import TripletLoss
from reid.trainers import CoTeaching
from reid.evaluators import Evaluator, extract_features
from reid.utils.data import transforms as T
import torch.nn.functional as F
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler
from reid.utils.serialization import load_checkpoint, save_checkpoint

from sklearn.cluster import DBSCAN
from reid.rerank import re_ranking


def calScores(clusters, labels):
    """
    compute pair-wise precision pair-wise recall
    """
    from scipy.special import comb
    if len(clusters) == 0:
        return 0, 0
    else:
        curCluster = []
        for curClus in clusters.values():
            curCluster.append(labels[curClus])
        TPandFP = sum([comb(len(val), 2) for val in curCluster])
        TP = 0
        for clusterVal in curCluster:
            for setMember in set(clusterVal):
                if sum(clusterVal == setMember) < 2: continue
                TP += comb(sum(clusterVal == setMember), 2)
        FP = TPandFP - TP
        # FN and TN
        TPandFN = sum([comb(labels.tolist().count(val), 2) for val in set(labels)])
        FN = TPandFN - TP
        # cal precision and recall
        precision, recall = TP / (TP + FP), TP / (TP + FN)
        fScore = 2 * precision * recall / (precision + recall)
        return precision, recall, fScore


def get_data(name, data_dir, height, width, batch_size,
             workers):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, num_val=0.1)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    # use all training and validation images in target dataset
    train_set = dataset.trainval
    num_classes = dataset.num_trainval_ids

    transformer = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalizer,
    ])

    extfeat_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, extfeat_loader, test_loader


def get_source_data(name, data_dir, height, width, batch_size,
                    workers):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, num_val=0.1)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    # use all training images on source dataset
    train_set = dataset.train
    num_classes = dataset.num_train_ids

    transformer = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalizer,
    ])

    extfeat_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, extfeat_loader


def calDis(qFeature, gFeature):  # 246s
    x, y = F.normalize(qFeature), F.normalize(gFeature)
    # x, y = qFeature, gFeature
    m, n = x.shape[0], y.shape[0]
    disMat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
             torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    disMat.addmm_(1, -2, x, y.t())
    return disMat.clamp_(min=1e-5)


def labelUnknown(knownFeat, allLab, unknownFeat):
    # allLab--label from known
    disMat = calDis(knownFeat, unknownFeat)
    labLoc = disMat.argmin(dim=0)
    return allLab[labLoc]


def labelNoise(feature, labels):
    # features and labels with -1
    noiseFeat, pureFeat = feature[labels == -1, :], feature[labels != -1, :]
    pureLabs = labels[labels != -1]  # no outliers
    unLab = labelUnknown(pureFeat, pureLabs, noiseFeat)
    labels[labels == -1] = unLab
    return labels.numpy()


def getCenter(features, labels):
    allCenter = {}
    features = features[labels != -1, :]
    labels = labels[labels != -1]
    for pid in set(labels):
        allCenter[pid] = torch.from_numpy(features[labels == pid, :].mean(axis=0)).unsqueeze(0)
    return torch.cat(list(allCenter.values()))


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # Create data loaders
    assert args.num_instances > 1, "num_instances should be greater than 1"
    assert args.batch_size % args.num_instances == 0, \
        'num_instances should divide batch_size'
    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else \
            (256, 128)

    # get source data
    src_dataset, src_extfeat_loader = \
        get_source_data(args.src_dataset, args.data_dir, args.height,
                        args.width, args.batch_size, args.workers)
    # get target data
    tgt_dataset, num_classes, tgt_extfeat_loader, test_loader = \
        get_data(args.tgt_dataset, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers)

    # Create model
    # Hacking here to let the classifier be the number of source ids
    if args.src_dataset == 'dukemtmc':
        model = models.create(args.arch, num_classes=632, pretrained=False)
        coModel = models.create(args.arch, num_classes=632, pretrained=False)
    elif args.src_dataset == 'market1501':
        model = models.create(args.arch, num_classes=676, pretrained=False)
        coModel = models.create(args.arch, num_classes=676, pretrained=False)
    elif args.src_dataset == 'msmt17':
        model = models.create(args.arch, num_classes=1041, pretrained=False)
        coModel = models.create(args.arch, num_classes=1041, pretrained=False)
    elif args.src_dataset == 'cuhk03':
        model = models.create(args.arch, num_classes=1230, pretrained=False)
        coModel = models.create(args.arch, num_classes=1230, pretrained=False)
    else:
        raise RuntimeError('Please specify the number of classes (ids) of the network.')

    # Load from checkpoint
    if args.resume:
        print('Resuming checkpoints from finetuned model on another dataset...\n')
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        coModel.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        raise RuntimeWarning('Not using a pre-trained model.')
    model = nn.DataParallel(model).cuda()
    coModel = nn.DataParallel(coModel).cuda()

    # Criterion
    criterion = [
        TripletLoss(args.margin, args.num_instances, isAvg=False, use_semi=False).cuda(),
        TripletLoss(args.margin, args.num_instances, isAvg=False, use_semi=False).cuda()
    ]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr
    )
    coOptimizer = torch.optim.Adam(
        coModel.parameters(), lr=args.lr
    )

    optims = [optimizer, coOptimizer]

    # training stage transformer on input images
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
        T.Resize((args.height, args.width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(), normalizer,
        T.RandomErasing(probability=0.5, sh=0.2, r1=0.3)
    ])

    # # Start training
    for iter_n in range(args.iteration):
        if args.lambda_value == 0:
            source_features = 0
        else:
            # get source datas' feature
            source_features, _ = extract_features(model, src_extfeat_loader, print_freq=args.print_freq, numStripe=None)
            # synchronization feature order with src_dataset.train
            source_features = torch.cat([source_features[f].unsqueeze(0) for f, _, _ in src_dataset.train], 0)

            # extract training images' features
        print('Iteration {}: Extracting Target Dataset Features...'.format(iter_n + 1))
        target_features, _ = extract_features(model, tgt_extfeat_loader, print_freq=args.print_freq, numStripe=None)
        # synchronization feature order with dataset.train
        target_features = torch.cat([target_features[f].unsqueeze(0) for f, _, _ in tgt_dataset.trainval], 0)
        # calculate distance and rerank result
        print('Calculating feature distances...')
        target_features = target_features.numpy()
        rerank_dist = re_ranking(source_features, target_features, lambda_value=args.lambda_value)
        if iter_n == 0:
            # DBSCAN cluster
            tri_mat = np.triu(rerank_dist, 1)  # tri_mat.dim=2
            tri_mat = tri_mat[np.nonzero(tri_mat)]  # tri_mat.dim=1
            tri_mat = np.sort(tri_mat, axis=None)
            top_num = np.round(args.rho * tri_mat.size).astype(int)
            eps = tri_mat[:top_num].mean()
            print('eps in cluster: {:.3f}'.format(eps))
            cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=8)
        # select & cluster images as training set of this epochs
        print('Clustering and labeling...')
        labels = cluster.fit_predict(rerank_dist)
        num_ids = len(set(labels)) - 1
        print('Iteration {} have {} training ids'.format(iter_n + 1, num_ids))
        # generate new dataset
        new_dataset = []
        # assign label for target ones
        newLab = labelNoise(torch.from_numpy(target_features), torch.from_numpy(labels))
        # unknownFeats = target_features[labels==-1,:]
        counter = 0
        from collections import defaultdict
        realIDs, fakeIDs = defaultdict(list), []
        for (fname, realID, cam), label in zip(tgt_dataset.trainval, newLab):
            # dont need to change codes in trainer.py _parsing_input function and sampler function after add 0
            new_dataset.append((fname, label, cam))
            realIDs[realID].append(counter)
            fakeIDs.append(label)
            counter += 1
        precision, recall, fscore = calScores(realIDs, np.asarray(fakeIDs))
        print('Iteration {} have {} training images'.format(iter_n + 1, len(new_dataset)))
        print(f'precision:{precision * 100}, recall:{100 * recall}, fscore:{100 * fscore}')
        train_loader = DataLoader(
            Preprocessor(new_dataset, root=tgt_dataset.images_dir, transform=train_transformer),
            batch_size=args.batch_size, num_workers=4,
            sampler=RandomIdentitySampler(new_dataset, args.num_instances),
            pin_memory=True, drop_last=True
        )
        trainer = CoTeaching(
            model, coModel, train_loader, criterion, optims
        )

        # Start training
        for epoch in range(args.epochs):
            trainer.train(epoch, remRate=0.2 + (0.8 / args.iteration) * (1 + iter_n))  # to at most 80%
        # test only
        evaluator = Evaluator(model, print_freq=args.print_freq)
        rank_score = evaluator.evaluate(test_loader, tgt_dataset.query, tgt_dataset.gallery)

    # Evaluate
    rank_score = evaluator.evaluate(test_loader, tgt_dataset.query, tgt_dataset.gallery)
    save_checkpoint({
        'state_dict': model.module.state_dict(),
        'epoch': epoch + 1, 'best_top1': rank_score.market1501[0],
    }, True, fpath=osp.join(args.logs_dir, 'adapted.pth.tar'))
    return rank_score.map, rank_score.market1501[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Triplet loss classification")
    # data
    parser.add_argument('--src_dataset', type=str, default='dukemtmc',
                        choices=datasets.names())
    parser.add_argument('--tgt_dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--noiseLam', type=float, default=0.5)
    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    parser.add_argument('--num_instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    # model
    parser.add_argument('--arch', type=str, default='resnet50',
                        choices=models.names())
    # loss
    parser.add_argument('--margin', type=float, default=0.5,
                        help="margin of the triplet loss, default: 0.5")
    parser.add_argument('--lambda_value', type=float, default=0.1,
                        help="balancing parameter, default: 0.1")
    parser.add_argument('--rho', type=float, default=1.6e-3,
                        help="rho percentage, default: 1.6e-3")
    # optimizer
    parser.add_argument('--lr', type=float, default=6e-5,
                        help="learning rate of all parameters")
    # training configs
    parser.add_argument('--resume', type=str, metavar='PATH',
                        default='')
    parser.add_argument('--evaluate', type=int, default=0,
                        help="evaluation only")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--iteration', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=30)
    # metric learning
    parser.add_argument('--dist_metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])
    # misc
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default='')
    parser.add_argument('--logs_dir', type=str, metavar='PATH',
                        default='')

    args = parser.parse_args()
    mean_ap, rank1 = main(args)
