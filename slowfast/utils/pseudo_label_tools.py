import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn.functional as F

def cal_threshold(x,THRE=2):
    top_prob, top_label = torch.topk(F.softmax(x, dim=1), k=1)
    top_label = top_label.squeeze().t()
    top_prob = top_prob.squeeze().t()
    if THRE == -1:
        threshold = -1 #selcet all samples
    else:
        top_mean, top_std = top_prob.mean(), top_prob.std()
        threshold = top_mean - THRE * top_std
    return top_label, top_prob, threshold