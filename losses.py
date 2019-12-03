"""
Losses
"""
import torch
from torch import nn

mse_loss = nn.MSELoss()


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
    def __init__(self, mask_flag=True, brightness_bias_flag=True):
        super(ProbaVLoss, self).__init__()
        self.mask_flag = mask_flag
        self.brightness_bias_flag = brightness_bias_flag

    def brightness_bias(self, predict, target, target_mask):
        return torch.mean(target.mul(target_mask) - predict.mul(target_mask))

    def _mse(self, predict, target):
        return torch.mean((predict - target) * (predict - target))

    def _mse_mask(self, predict, target, target_mask):
        if self.brightness_bias_flag:
            bb = self.brightness_bias(predict, target, target_mask)
            predict_out = predict + bb
        else:
            predict_out = predict
        return mse_loss(predict_out.mul(target_mask), target.mul(target_mask))

    def _mad_mask(self, predict, target, target_mask):
        return torch.mean(torch.abs(predict.mul(target_mask) - target.mul(target_mask)))

    def forward(self, output, target, target_mask):
        """
        original image and output mask is lo rez
        final output of model is hi rez

        """
        if self.mask_flag:
            loss_mse = self._mse_mask(output, target, target_mask)
            loss_mad = self._mad_mask(output, target, target_mask)
            loss = loss_mse + 5 * loss_mad
        else:
            loss = nn.MSELoss(output, target)
        return loss


class ProbaVEval(nn.Module):
    def __init__(self, mask_flag=True):
        super(ProbaVEval, self).__init__()
        self.mask_flag = mask_flag

    def brightness_bias(self, predict, target, target_mask):
        return torch.mean(target.mul(target_mask) - predict.mul(target_mask))

    def c_mse(self, predict, target, target_mask):
        predict = predict + self.brightness_bias(predict, target, target_mask)
        predict = predict.mul(target_mask)
        target = target.mul(target_mask)
        return mse_loss(target, predict)

    def c_psnr(self, cmse):
        return -10 * torch.log10(cmse)

    def forward(self, output, target, target_mask, baseline=None):
        """
        Calculate evaluation according to scoring on competition website

        """
        cMSE = self.c_mse(output, target, target_mask)
        score = self.c_psnr(cMSE)
        if baseline is not None:
            score = baseline / score
        return score
