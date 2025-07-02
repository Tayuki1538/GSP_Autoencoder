import torch
import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

def mean_squared_error(output, target):
    return F.mse_loss(output, target)

def l1_loss(output, target):
    return F.l1_loss(output, target)

def bce_loss(output, target):
    return F.binary_cross_entropy(output, target)

def bce_with_logits_loss(output, target):
    pos_weight = torch.tensor([24])
    pos_weight = pos_weight.cuda() if torch.cuda.is_available() else pos_weight
    return F.binary_cross_entropy_with_logits(output, target, pos_weight=pos_weight)    # 大体 pos_weight = negative_label / positive_label