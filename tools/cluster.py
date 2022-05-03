
import ipdb
import numpy as np
import os
import pickle
import torch
from iopath.common.file_io import g_pathmgr

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TestMeter
import torch.nn.functional as F


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
        
    return res

@torch.no_grad()
def k_means(target_features, target_targets, target_center, cfg, logger):
    
    target_center = target_center.data.clone()
    best_prec = 0.0
    eps = 1e-6
    cluster_iter =5

    for itr in range(cluster_iter):
        dist_xt_ct_temp = target_features.unsqueeze(1) - target_center.unsqueeze(0)
        dist_xt_ct = dist_xt_ct_temp.pow(2).sum(2)
        _, idx_sim = (-1 * dist_xt_ct).data.topk(1, 1, True, True)

        prec1 = accuracy(-1 * dist_xt_ct.data, target_targets, topk=(1,))[0].item()
        logger.info("!!!!!!!!!!!!!!{} kmeans target domain acc:{}.".format(itr,prec1))
        is_best = prec1 > best_prec
        if is_best:
            best_prec = prec1
        
        target_center_temp = torch.cuda.FloatTensor(cfg.MODEL.NUM_CLASSES, target_center.size(1)).fill_(0)
        count = torch.cuda.FloatTensor(cfg.MODEL.NUM_CLASSES, 1).fill_(0) 

        for k in range(cfg.MODEL.NUM_CLASSES):
            target_center_temp[k] += target_features[idx_sim.squeeze(1) == k].sum(0)
            count[k] += (idx_sim.squeeze(1) == k).float().sum()

        target_center_temp /= (count + eps)
        target_center = target_center_temp.clone()
        torch.cuda.empty_cache()
    
    torch.cuda.empty_cache()

    return target_center



@torch.no_grad()
def get_center(cenval_source_loader, cenval_target_loader, model, source_center, target_center, cfg, logger, cur_epoch, writer, init=True,retpseudo=False):
    
    model.eval()

    if init:
        logger.info("init get cluster center from source target labels.")
        #start init center, need labels
        source_center_sum = torch.cuda.FloatTensor(cfg.MODEL.NUM_CLASSES, 512).fill_(0)
        target_center_sum = torch.cuda.FloatTensor(cfg.MODEL.NUM_CLASSES, 512).fill_(0)
        
        source_count_class = torch.cuda.FloatTensor(cfg.MODEL.NUM_CLASSES, 1).fill_(0)
        target_count_class = torch.cuda.FloatTensor(cfg.MODEL.NUM_CLASSES, 1).fill_(0)

        new_source_center = torch.cuda.FloatTensor(cfg.MODEL.NUM_CLASSES, 512).fill_(0)
        new_target_center = torch.cuda.FloatTensor(cfg.MODEL.NUM_CLASSES, 512).fill_(0)


        for cur_iter, (inputs, labels, video_idx, _) in enumerate(cenval_source_loader):
            if cfg.NUM_GPUS:
                # Transfer the data to the current GPU device.
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
                # Transfer the data to the current GPU device.

                labels = labels.cuda()
                labels = F.one_hot(labels, num_classes=cfg.MODEL.NUM_CLASSES)
                video_idx = video_idx.cuda()

            features, _ = model.module.forward_features_logits(inputs)
            # source_dataset_features[video_idx.cuda(), :] = features.data.clone()
            source_center_sum +=  (features.unsqueeze(1) * labels.unsqueeze(2)).sum(0) #累加batch数据
            source_count_class += labels.sum(0).unsqueeze(1)
        source_center_sum = du.all_reduce(source_center_sum,average=False)
        source_count_class = du.all_reduce(source_count_class,average=False)
        new_source_center = source_center_sum/source_count_class
    
        for cur_iter, (inputs, labels, video_idx, _) in enumerate(cenval_target_loader):
            if cfg.NUM_GPUS:
                # Transfer the data to the current GPU device.
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
                # Transfer the data to the current GPU device.

                labels = labels.cuda()
                labels = F.one_hot(labels, num_classes=cfg.MODEL.NUM_CLASSES)

                video_idx = video_idx.cuda()

            features, _ = model.module.forward_features_logits(inputs)
            # source_dataset_features[video_idx.cuda(), :] = features.data.clone()
            target_center_sum +=  (features.unsqueeze(1) * labels.unsqueeze(2)).sum(0) #累加batch数据
            target_count_class += labels.sum(0).unsqueeze(1)
        target_center_sum = du.all_reduce(target_center_sum,average=False)
        target_count_class = du.all_reduce(target_count_class,average=False)
        new_target_center = target_center_sum/target_count_class
        logger.info("over.")
        torch.cuda.empty_cache()
        return new_source_center.detach(),new_target_center.detach()
        

    else:
        logger.info("epoch update cluster center.")
        #based on  target samples features and pseudo labels，centers  ---> kmeans 

        source_center_sum = torch.cuda.FloatTensor(cfg.MODEL.NUM_CLASSES, 512).fill_(0)
        source_count_class = torch.cuda.FloatTensor(cfg.MODEL.NUM_CLASSES, 1).fill_(0)
        new_source_center = torch.cuda.FloatTensor(cfg.MODEL.NUM_CLASSES, 512).fill_(0)


        for cur_iter, (inputs, labels, video_idx, _) in enumerate(cenval_source_loader):
            if cfg.NUM_GPUS:
                # Transfer the data to the current GPU device.
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
                # Transfer the data to the current GPU device.

                labels = labels.cuda()
                labels = F.one_hot(labels, num_classes=cfg.MODEL.NUM_CLASSES)
                video_idx = video_idx.cuda()

            features, _ = model.module.forward_features_logits(inputs)
            # source_dataset_features[video_idx.cuda(), :] = features.data.clone()
            source_center_sum +=  (features.unsqueeze(1) * labels.unsqueeze(2)).sum(0) #累加batch数据
            source_count_class += labels.sum(0).unsqueeze(1)

        source_center_sum = du.all_reduce(source_center_sum,average=False)
        source_count_class = du.all_reduce(source_count_class,average=False)
        new_source_center = source_center_sum/source_count_class

        all_pse_labels = torch.cuda.LongTensor(len(cenval_target_loader.dataset)).fill_(0)
        # eval the cluster effect
        all_target_features = torch.cuda.FloatTensor(len(cenval_target_loader.dataset),512).fill_(0)

        target_center_sum = torch.cuda.FloatTensor(cfg.MODEL.NUM_CLASSES, 512).fill_(0)
        target_count_class = torch.cuda.FloatTensor(cfg.MODEL.NUM_CLASSES, 1).fill_(0)

        new_target_center = torch.cuda.FloatTensor(cfg.MODEL.NUM_CLASSES, 512).fill_(0)

        for cur_iter, (inputs, labels, video_idx, _) in enumerate(cenval_target_loader):
            if cfg.NUM_GPUS:
                # Transfer the data to the current GPU device.
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
                # Transfer the data to the current GPU device.

                labels = labels.cuda()
                labels = F.one_hot(labels, num_classes=cfg.MODEL.NUM_CLASSES)
                video_idx = video_idx.cuda()


            features, logits = model.module.forward_features_logits(inputs)
            all_target_features[video_idx.cuda(),:] = features

            pred = logits.data.max(1)[1]
            all_pse_labels[video_idx.cuda()] = pred

            target_center_sum +=  (features.unsqueeze(1) * F.one_hot(pred, num_classes=cfg.MODEL.NUM_CLASSES).unsqueeze(2)).sum(0) #累加batch数据
            target_count_class += F.one_hot(pred, num_classes=cfg.MODEL.NUM_CLASSES).sum(0).unsqueeze(1)
        
        all_target_features = du.all_reduce(all_target_features,average=False)
        all_pse_labels = du.all_reduce(all_pse_labels,average=False)
        target_center_sum = du.all_reduce(target_center_sum,average=False)
        target_count_class = du.all_reduce(target_count_class,average=False)

        new_target_center = target_center_sum/target_count_class

        # new_target_center = k_means(all_target_features,all_pse_labels,new_target_center,cfg,logger)
        logger.info("over.")
        torch.cuda.empty_cache()
        if retpseudo:
            return new_source_center.detach(),new_target_center.detach(),all_pse_labels.detach()
        else:
            return new_source_center.detach(),new_target_center.detach()

        
        
    

        
        


    




        



