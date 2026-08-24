import torch
import torch.nn as nn

from ..builder import LOSSES


def _reduce_loss(loss, reduction="mean", avg_factor=None):
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        if avg_factor is not None:
            return loss.sum() / avg_factor
        return loss.mean()
    return loss


@LOSSES.register_module()
class GaussianFocalLoss(nn.Module):
    def __init__(
        self,
        alpha=2.0,
        gamma=4.0,
        reduction="mean",
        loss_weight=1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs,
    ):
        del kwargs
        reduction = reduction_override if reduction_override else self.reduction

        pred = pred.sigmoid()
        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()
        neg_weights = (1 - target).pow(self.gamma)

        pos_loss = -(pred.clamp(min=1e-12).log()) * (1 - pred).pow(self.alpha) * pos_mask
        neg_loss = -((1 - pred).clamp(min=1e-12).log()) * pred.pow(self.alpha) * neg_weights * neg_mask
        loss = pos_loss + neg_loss

        if weight is not None:
            loss = loss * weight

        loss = _reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)
        return self.loss_weight * loss
