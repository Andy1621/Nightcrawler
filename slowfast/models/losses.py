#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetCrossEntropy(nn.Module):
    """
    Cross entropy loss with soft target.
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(SoftTargetCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        loss = torch.sum(-y * F.log_softmax(x, dim=-1), dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError

class KLLoss(nn.Module):
    def __init__(self,T, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()
        # print('=========using KL Loss=and has temperature and * bz==========')
        self.error_metric = error_metric
        self.T = T

    def forward(self, prediction, label):

        batch_size = prediction.shape[0]
        # probs1 = F.log_softmax(prediction,dim=-1)
        probs1 = F.log_softmax(prediction/self.T,dim=-1)
        # KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
        #                      F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
        label = label.to(torch.float16)/self.T
        # print("probs1",type(probs1),probs1.size(),probs1)
        # label = torch.log(label)
        # print("label",type(label),label.size(),label)
        # probs2 = F.softmax(label * 10, 1)
        loss = self.error_metric(probs1, label)* self.T * self.T
        return loss

def TarDisClusterLoss(epoch, output, target, softmax=True, em=True): #em=False
    if softmax:
        prob_p = F.softmax(output, dim=1)
    else:
        prob_p = output / output.sum(1, keepdim=True)
    if em:
        prob_q = prob_p
    else:
        prob_q1 = Variable(torch.cuda.FloatTensor(prob_p.size()).fill_(0))
        prob_q1.scatter_(1, target.unsqueeze(1), torch.ones(prob_p.size(0), 1).cuda()) # assigned pseudo labels
        if  epoch == 0:
            prob_q = prob_q1
        else:
            prob_q2 = prob_p / prob_p.sum(0, keepdim=True).pow(0.5)
            prob_q2 /= prob_q2.sum(1, keepdim=True)
            prob_q = 0.5 * prob_q1 + 0.5 * prob_q2
    
    if softmax:
        loss = - (prob_q * F.log_softmax(output, dim=1)).sum(1).mean()
    else:
        loss = - (prob_q * prob_p.log()).sum(1).mean()
    
    return loss

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "soft_cross_entropy": SoftTargetCrossEntropy,
    "KL": KLLoss,
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]
