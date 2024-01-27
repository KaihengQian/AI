import torch
import torch.nn as nn


class ImageAlexNet(nn.Module):
    def __init__(self, output_dim, dropout=0.0):
        super(ImageAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


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


class ImageMobileNetV1(nn.Module):
    def __init__(self, output_dim, dropout=0.0):
        super(ImageMobileNetV1, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
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

        self.fc = nn.Linear(1024, output_dim)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
