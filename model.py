"""Inception-v3 model for dog breed classification."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, out_3x3_reduce, out_3x3,
                 out_5x5_reduce, out_5x5, out_pool):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, out_1x1, 1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_3x3_reduce, 1),
            nn.Conv2d(out_3x3_reduce, out_3x3, 3, padding=1))
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_5x5_reduce, 1),
            nn.Conv2d(out_5x5_reduce, out_5x5, 5, padding=2))
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool, 1))
    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x),
                         self.branch3(x), self.branch4(x)], 1)
