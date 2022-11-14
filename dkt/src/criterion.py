import torch.nn as nn


def get_criterion(pred, target):
    loss = nn.BCEWithLogitsLoss(reduction="none")
    return loss(pred, target)
