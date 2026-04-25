import torch
import torch.nn as nn
import torch.nn.functional as F

BN_MOMENTUM = 0.1

#Residual Block

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)

        # adjust residual if needed
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


# Simple High Resolution Net Module (Paper core model Idea)
class SimpleHRModule(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # branch 1 (high resolution)
        self.branch1 = BasicBlock(channels, channels)

        # branch 2 (low resolution)
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            BasicBlock(channels, channels)
        )

        # fuse layers
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.down = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        x1 = self.branch1(x)          # high-res
        x2 = self.branch2(x)          # low-res

        # fuse
        x1_fused = x1 + self.up(x2)
        x2_fused = x2 + self.down(x1)

        return x1_fused, x2_fused



# Final Model

class SimpleHRNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # HR modules
        self.hr1 = SimpleHRModule(64)
        self.hr2 = SimpleHRModule(64)

        # Final head
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)

        x1, x2 = self.hr1(x)
        x1, x2 = self.hr2(x1)   # keep high-res path dominant

        x = self.final_conv(x1)

        # global average pooling
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)

        x = self.classifier(x)
        return x