import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inc, outc, 3, padding=1),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True),
            nn.Conv2d(outc, outc, 3, padding=1),
            nn.BatchNorm2d(outc),
        )
        self.skip = nn.Conv2d(inc, outc, 1) if inc != outc else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))


class ResUNet(nn.Module):
    def __init__(self, ch=4):
        super().__init__()
        self.e1 = Block(1, ch)
        self.e2 = Block(ch, ch * 2)
        self.e3 = Block(ch * 2, ch * 4)
        self.pool = nn.MaxPool2d(2)
        self.bn = Block(ch * 4, ch * 8)
        self.d3 = Block(ch * 8 + ch * 4, ch * 4)
        self.d2 = Block(ch * 4 + ch * 2, ch * 2)
        self.d1 = Block(ch * 2 + ch, ch)
        self.out = nn.Sequential(nn.Conv2d(ch, 1, 1), nn.Tanh())

    def _up(self, x, target):
        return F.interpolate(x, size=target.shape[2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        b = self.bn(self.pool(s3))
        x = self.d3(torch.cat([self._up(b, s3), s3], dim=1))
        x = self.d2(torch.cat([self._up(x, s2), s2], dim=1))
        x = self.d1(torch.cat([self._up(x, s1), s1], dim=1))
        return self.out(x)
