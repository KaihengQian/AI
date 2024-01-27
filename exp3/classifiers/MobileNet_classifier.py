import torch
import torch.nn as nn


# 定义深度可分离卷积层
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dropout):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1,
                                        groups=in_channels, bias=False)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)  # 添加Dropout层

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=10, dropout=0.0):
        super(MobileNetV1, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            DepthwiseSeparableConv(32, 64, stride=1, dropout=dropout),
            DepthwiseSeparableConv(64, 128, stride=2, dropout=dropout),
            DepthwiseSeparableConv(128, 128, stride=1, dropout=dropout),
            DepthwiseSeparableConv(128, 256, stride=2, dropout=dropout),
            DepthwiseSeparableConv(256, 256, stride=1, dropout=dropout),

            DepthwiseSeparableConv(256, 512, stride=2, dropout=dropout),
            DepthwiseSeparableConv(512, 512, stride=1, dropout=dropout),
            DepthwiseSeparableConv(512, 512, stride=1, dropout=dropout),
            DepthwiseSeparableConv(512, 512, stride=1, dropout=dropout),
            DepthwiseSeparableConv(512, 512, stride=1, dropout=dropout),

            DepthwiseSeparableConv(512, 1024, stride=2, dropout=dropout),
            DepthwiseSeparableConv(1024, 1024, stride=1, dropout=dropout),

            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
