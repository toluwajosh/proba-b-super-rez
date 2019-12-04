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
    def __init__(self, mask_flag=True, brightness_bias_flag=True, crop_size=3):
        super(ProbaVLoss, self).__init__()
        self.mask_flag = mask_flag
        self.brightness_bias_flag = brightness_bias_flag
        self.crop_size = crop_size

    def brightness_bias(self, predict, target):
        return torch.mean(target - predict)

    def _mse(self, predict, target):
        return mse_loss(predict, target)

    def _mse_b(self, predict, target):
        if self.brightness_bias_flag:
            bb = self.brightness_bias(predict, target)
            predict_out = predict + bb
        else:
            predict_out = predict
        return mse_loss(predict_out, target)

    def _ssim(self, predict, target):
        # return torch.mean(torch.abs(predict - target)) # _mad
        return torch.mean(dssim(predict, target))

    def _full_loss(self, predict, target):
        loss_mse = self._mse_b(predict, target)
        loss_ssim = self._ssim(predict, target)
        return loss_mse + 5 * loss_ssim

    def _cropped_loss(self, predict, target):
        cropped_target = target[
            :, :, self.crop_size : -self.crop_size, self.crop_size : -self.crop_size
        ]
        max_crop = self.crop_size * 2
        min_loss = 999999999
        for u in range(max_crop):
            for v in range(max_crop):
                cropped_predict = predict[:, :, u:384-max_crop+u, v:384-max_crop+v]
                loss_mse = self._mse_b(cropped_predict, cropped_target)
                loss_ssim = self._ssim(cropped_predict, cropped_target)
                loss = loss_mse + 5 * loss_ssim
                if loss < min_loss:
                    min_loss = loss
        return min_loss

    def forward(self, predict, target, target_mask):
        if self.mask_flag:
            target = target.mul(target_mask)
            predict = predict.mul(target_mask)
        if self.crop_size:
            loss = self._cropped_loss(predict, target)
        else:
            loss = self._full_loss(predict, target)
        return loss
        


class ProbaVEval(nn.Module):
    def __init__(self, mask_flag=True, crop_size=3):
        super(ProbaVEval, self).__init__()
        self.mask_flag = mask_flag
        self.crop_size = crop_size

    def brightness_bias(self, predict, target):
        return torch.mean(target - predict)

    def c_mse(self, predict, target):
        bb = self.brightness_bias(predict, target)
        predict_out = predict + bb
        return mse_loss(predict_out, target)
    
    def cropped_cmse(self, predict, target):
        cropped_target = target[
            :, :, self.crop_size : -self.crop_size, self.crop_size : -self.crop_size
        ]
        max_crop = self.crop_size * 2
        min_loss = 999999999
        for u in range(max_crop):
            for v in range(max_crop):
                cropped_predict = predict[:, :, u:384-max_crop+u, v:384-max_crop+v]
                loss = self.c_mse(cropped_predict, cropped_target)
                if loss < min_loss:
                    min_loss = loss
        return min_loss

    def c_psnr(self, cmse):
        return -10 * torch.log10(cmse)

    def forward(self, predict, target, target_mask, baseline=None):
        """
        Calculate evaluation according to scoring on competition website

        """
        target = target.mul(target_mask)
        predict = predict.mul(target_mask)
        cMSE = self.cropped_cmse(predict, target)
        score = self.c_psnr(cMSE)
        if baseline is not None:
            score = baseline / score
        return score


def dssim(x, y):
    """ Official implementation
    def SSIM(self, x, y):
        C1 = 0.01 ** 2 # why not use L=255
        C2 = 0.03 ** 2 # why not use L=255
        mu_x = slim.avg_pool2d(x, 3, 1, 'SAME')
        mu_y = slim.avg_pool2d(y, 3, 1, 'SAME')
        # if this implementatin equvalent to the SSIM paper?
        sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'SAME') - mu_x ** 2 
        sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'SAME') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'SAME') - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d
        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)
    """
    avepooling2d = torch.nn.AvgPool2d(3, stride=1, padding=[1, 1])
    mu_x = avepooling2d(x)
    mu_y = avepooling2d(y)
    # sigma_x = avepooling2d((x-mu_x)**2)
    # sigma_y = avepooling2d((y-mu_y)**2)
    # sigma_xy = avepooling2d((x-mu_x)*(y-mu_y))
    sigma_x = avepooling2d(x ** 2) - mu_x ** 2
    sigma_y = avepooling2d(y ** 2) - mu_y ** 2
    sigma_xy = avepooling2d(x * y) - mu_x * mu_y
    k1_square = 0.01 ** 2
    k2_square = 0.03 ** 2
    # L_square = 255**2
    L_square = 1
    SSIM_n = (2 * mu_x * mu_y + k1_square * L_square) * (
        2 * sigma_xy + k2_square * L_square
    )
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + k1_square * L_square) * (
        sigma_x + sigma_y + k2_square * L_square
    )
    SSIM = SSIM_n / SSIM_d
    return torch.clamp((1 - SSIM) / 2, 0, 1)
