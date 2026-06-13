import torch
import torch.nn.functional as F


def ssim_loss(pred, target, window=11):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    pad = window // 2
    mu1 = F.avg_pool2d(pred, window, stride=1, padding=pad)
    mu2 = F.avg_pool2d(target, window, stride=1, padding=pad)
    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    mu12 = mu1 * mu2
    s1 = F.avg_pool2d(pred * pred, window, stride=1, padding=pad) - mu1_sq
    s2 = F.avg_pool2d(target * target, window, stride=1, padding=pad) - mu2_sq
    s12 = F.avg_pool2d(pred * target, window, stride=1, padding=pad) - mu12
    num = (2 * mu12 + c1) * (2 * s12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (s1 + s2 + c2)
    return 1 - (num / den).mean()


def phys_penalty(pred, ct):
    scatter = ct - pred
    return F.relu(-scatter).mean()


def total_loss(pred, ct):
    l1 = F.l1_loss(pred, ct)
    sl = ssim_loss(pred, ct)
    pp = phys_penalty(pred, ct)
    return l1 + 0.5 * sl + 0.1 * pp
