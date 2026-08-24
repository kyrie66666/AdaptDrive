import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        use_sigmoid=False,
        use_mask=False,
        reduction="mean",
        class_weight=None,
        loss_weight=1.0,
    ):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.class_weight = class_weight
        self.loss_weight = loss_weight

    def forward(
        self,
        pred,
        label,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs,
    ):
        del kwargs
        reduction = reduction_override if reduction_override else self.reduction

        if self.use_sigmoid:
            loss = F.binary_cross_entropy_with_logits(
                pred, label, weight=weight, reduction="none"
            )
        else:
            class_weight = None
            if self.class_weight is not None:
                class_weight = pred.new_tensor(self.class_weight)
            loss = F.cross_entropy(
                pred,
                label.long(),
                weight=class_weight,
                reduction="none",
            )
            if weight is not None:
                loss = loss * weight

        if reduction == "mean":
            if avg_factor is not None:
                loss = loss.sum() / avg_factor
            else:
                loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return self.loss_weight * loss
