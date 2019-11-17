"""
Losses
"""
import torch
from torch import nn


class MSEMask(nn.Module):
    def __init__(self, mask_flag=True):
        super(MSEMask, self).__init__()
        self.mask_flag = mask_flag

    def forward(self, output, target, output_mask, target_mask):
        if self.mask_flag:
            output = output.mul(output_mask)
            target = target.mul(target_mask)
            loss = torch.mean((output - target) * (output - target))
        else:
            loss = nn.MSELoss(output, target)
        return loss


class ProbaVLoss(nn.Module):
    def __init__(self, mask_flag=True):
        super(ProbaVLoss, self).__init__()
        self.mask_flag = mask_flag

    def _mse(self, predict, target):
        return torch.mean((predict - target) * (predict - target))

    def forward(self, output_lo, output, target, output_mask, target_mask):
        """
        original image and output mask is lo rez
        final output of model is hi rez

        """
        if self.mask_flag:
            # lo_rez loss
            # print("output_lo.shape: ", output_lo.shape)
            # print("output.shape: ", output.shape)
            # print("target.shape: ", target.shape)
            # print("output_mask.shape: ", output_mask.shape)
            # print("target_mask.shape: ", target_mask.shape)
            # exit(0)
            mask_weight = 0.85
            target_lo = torch.nn.functional.interpolate(target, output_lo.shape[2:])
            target_lo_mask = torch.nn.functional.interpolate(
                target_mask, output_lo.shape[2:]
            )
            lo_rez_loss = self._mse(output_lo, target_lo)
            # output_lo = output_lo.mul(output_mask)
            target_lo = target_lo.mul(target_lo_mask)
            lo_rez_loss = (1 - mask_weight) * lo_rez_loss + mask_weight * self._mse(
                output_lo, target_lo
            )

            # hi_rez loss
            # output_mask_hi = torch.nn.functional.interpolate(
            #     output_mask, output.shape[2:]
            # )
            hi_rez_loss = self._mse(output, target)
            # output = output.mul(output_mask_hi)
            target = target.mul(target_mask)
            hi_rez_loss = (1 - mask_weight) * hi_rez_loss + mask_weight * self._mse(
                output, target
            )

            loss_weight = 0.75
            loss = loss_weight * lo_rez_loss + (1 - loss_weight) * hi_rez_loss
        else:
            loss = nn.MSELoss(output, target)
        return loss
