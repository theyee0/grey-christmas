import torch
from torch.data.utils import DataLoader
from torch import nn
from torch.optim import Adam

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, downsample=None stride=1):
        super(Bottleneck, self).__init__()

        self.convolve0 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

        self.convolve1 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

        self.convolve2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(out_channels),
        )

        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        initial = x

        x = self.convolve0(x)
        x = self.convolve1(x)
        x = self.convolve2(x)

        if self.downsample:
            x = self.downsample(x)

        return self.relu(initial + x)


class Forecast(nn.Module):
    in_channels = 64

    def __init__(self, block_sizes, classes, channels=1):
        super(Forecast, self).__init__()

        self.convolve0 = nn.Conv1d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm0 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.max_pool = MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer0 = self._make_layer(block_sizes[0], channels=64)
        self.layer1 = self._make_layer(block_sizes[1], channels=128, stride=2)
        self.layer2 = self._make_layer(block_sizes[2], channels=256, stride=2)
        self.layer3 = self._make_layer(block_sizes[3], channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fully_connected = nn.Linear(512 * BottleNeck.expansion, classes)

    def _make_layer(self, blocks, channels, stride=1):
        target_channels = channels * BottleNeck.expansion

        downsampler = None

        if stride != 1 or self.in_channels != target_channels:
            downsampler = nn.Sequential(
                nn.Conv1d(self.in_channels, target_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(target_channels)
            )

        layers = [BottleNeck(self.in_channels, channels, downsample=downsampler, stride=stride)]

        for i in range(blocks - 1):
            layers.append(BottleNeck(self.in_channels, channels))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.convolve0(x)
        x = self.batch_norm0(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)

        x = self.fully_connected(x)

        return x
