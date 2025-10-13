import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Basic Loss Components
# -----------------------------
class IoULoss(nn.Module):
    """Intersection-over-Union loss (1 - IoU)"""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        intersection = (preds * targets).sum(dim=(1,2,3))
        union = (preds + targets - preds * targets).sum(dim=(1,2,3))
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou.mean()


class DiceLoss(nn.Module):
    """1 - Dice coefficient"""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        intersection = (preds * targets).sum(dim=(1,2,3))
        dice = (2. * intersection + self.smooth) / (preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) + self.smooth)
        return 1 - dice.mean()


class BalancedBCELoss(nn.Module):
    """Balanced BCE for imbalanced foreground/background"""
    def __init__(self, pos_weight=1.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, preds, targets):
        bce = F.binary_cross_entropy_with_logits(preds, targets, pos_weight=torch.tensor(self.pos_weight, device=preds.device))
        return bce


class BoundaryLoss(nn.Module):
    """Boundary-weighted loss using distance maps (distance maps should be precomputed and aligned with targets)"""
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets, distance_map):
        preds = torch.sigmoid(preds)
        bce = F.binary_cross_entropy(preds, targets, reduction='none')
        weighted_bce = (bce * distance_map).mean()
        return weighted_bce


# -----------------------------
# Task-Specific Combined Losses
# -----------------------------
class CamouflagedObjectLoss(nn.Module):
    """BCE + IoU"""
    def __init__(self, w_bce=0.5, w_iou=0.5):
        super().__init__()
        self.w_bce = w_bce
        self.w_iou = w_iou
        self.bce = nn.BCEWithLogitsLoss()
        self.iou = IoULoss()

    def forward(self, preds, targets):
        loss_bce = self.bce(preds, targets)
        loss_iou = self.iou(preds, targets)
        return self.w_bce * loss_bce + self.w_iou * loss_iou


class ShadowDetectionLoss(nn.Module):
    """Balanced BCE"""
    def __init__(self, pos_weight=1.5):
        super().__init__()
        self.loss = BalancedBCELoss(pos_weight=pos_weight)

    def forward(self, preds, targets):
        return self.loss(preds, targets)


class StructureSegmentationLoss(nn.Module):
    """BCE + DICE + Boundary weighting"""
    def __init__(self, w_bce=0.4, w_dice=0.4, w_boundary=0.2):
        super().__init__()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.w_boundary = w_boundary
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss()

    def forward(self, preds, targets, distance_map):
        loss_bce = self.bce(preds, targets)
        loss_dice = self.dice(preds, targets)
        loss_boundary = self.boundary(preds, targets, distance_map)
        return (self.w_bce * loss_bce +
                self.w_dice * loss_dice +
                self.w_boundary * loss_boundary)