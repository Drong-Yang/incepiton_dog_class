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


class InceptionV3(nn.Module):
    def __init__(self, num_classes=120):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.inception_blocks = nn.Sequential(
            InceptionBlock(64, 64, 96, 128, 16, 32, 32),
            InceptionBlock(256, 128, 128, 192, 32, 96, 64),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(480, num_classes)
    def forward(self, x):
        x = self.features(x)
        x = self.inception_blocks(x)
        x = self.avgpool(x).flatten(1)
        return self.classifier(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


def create_model(model_name='inception', num_classes=120, pretrained=False):
    if model_name == 'inception':
        return InceptionV3(num_classes)
    elif model_name == 'resnet':
        from torchvision.models import resnet50
        model = resnet50(pretrained=pretrained)
        model.fc = nn.Linear(2048, num_classes)
        return model
    raise ValueError(f'Unknown model: {model_name}')
