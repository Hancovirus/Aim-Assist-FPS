import torch
import config
from torch import nn as nn
from torch.nn import functional as F
from utils import get_iou, bbox_attr


class SumSquaredErrorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l_coord = 12  # Hệ số điều chỉnh tổn thất tọa độ
        self.l_noobj = 0.25  # Hệ số điều chỉnh tổn thất no-object

    def forward(self, preds, targets):
        """
        Args:
            preds (tensor): Dự đoán từ mô hình (batch_size, S, S, 5B + C).
            targets (tensor): Ground truth (batch_size, S, S, 5B + C).
        Returns:
            loss (tensor): Tổng tổn thất.
        """
        iou = get_iou(preds, targets)                 
        max_iou = torch.max(iou, dim=-1)[0]   
        C = config.C
        bbox_mask = bbox_attr(targets, 0) > 0.0
        p_template = bbox_attr(preds, 0) > 0.0
        obj_i = bbox_mask[..., 0:1]        
        responsible = torch.zeros_like(p_template).scatter_(      
            -1,
            torch.argmax(max_iou, dim=-1, keepdim=True),             
            value=1                         
        )
        obj_ij = obj_i * responsible       
        noobj_ij=~obj_ij
        # Class probability loss
        x_losses = F.mse_loss(
            obj_ij * bbox_attr(preds, 1),
            obj_ij * bbox_attr(targets, 1),
            reduction="sum"
        )
        y_losses = F.mse_loss(
            obj_ij * bbox_attr(preds, 2),
            obj_ij * bbox_attr(targets, 2),
            reduction="sum"
        )
        pos_losses = x_losses + y_losses
        p_width = bbox_attr(preds, 3)
        a_width = bbox_attr(targets, 3)
        width_losses = F.mse_loss(
            obj_ij * torch.sign(p_width) * (torch.sqrt(torch.abs(p_width) + config.EPSILON)),
            obj_ij * (torch.sqrt(a_width)),
            reduction="sum"
        )
        p_height = bbox_attr(preds, 4)
        a_height = bbox_attr(targets, 4)
        height_losses = F.mse_loss(
            obj_ij * torch.sign(p_height) * (torch.sqrt(torch.abs(p_height) + config.EPSILON)),
            obj_ij * (torch.sqrt(a_height)),
            reduction="sum"
        )
        dim_losses = width_losses + height_losses
        obj_confidence_losses = F.mse_loss(
            obj_ij * bbox_attr(preds, 0),
            obj_ij * torch.ones_like(max_iou),
            reduction="sum"
        )
        # print('obj_confidence_losses', obj_confidence_losses.item())
        noobj_confidence_losses = F.mse_loss(
            noobj_ij * bbox_attr(preds, 0),
            torch.zeros_like(max_iou),
            reduction="sum"
        )
        class_losses = F.mse_loss(
            obj_i * preds[..., :config.C],
            obj_i * targets[..., :config.C]
        )
        #print('noobj_confidence_losses',self.l_noobj * noobj_confidence_losses)
        #print('dim_losses',self.l_coord * dim_losses)
        #print('pos_losses', 3 * pos_losses)
        #print('obj_confidence_losses', obj_confidence_losses)
        # Tổng loss
        total = self.l_coord * (dim_losses) + 3 * pos_losses + obj_confidence_losses \
                + self.l_noobj * noobj_confidence_losses + class_losses
        return total / config.BATCH_SIZE