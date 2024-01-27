import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# 定义过渡层
class _Transition(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))

        self.add_module('conv', nn.Conv2d(in_channels, out_channels, 1, bias=False))
        self.add_module('pool', nn.AvgPool2d(2, 2))


# 定义稠密层
class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bottleneck_size, dropout):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, growth_rate * bottleneck_size, 1, bias=False)

        self.norm2 = nn.BatchNorm2d(growth_rate * bottleneck_size)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(growth_rate * bottleneck_size, growth_rate, 3, padding=1, bias=False)

        self.dropout = nn.Dropout(p=dropout)  # 添加Dropout层

    def forward(self, x):
        x = [x] if torch.is_tensor(x) else x
        x = self.conv1(self.relu1(self.norm1(torch.cat(x, 1))))
        x = self.conv2(self.relu2(self.norm2(x)))
        output = self.dropout(x)

        return output


# 定义稠密块
class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, in_channels, growth_rate, bottleneck_size, dropout):
        super().__init__()

        for i in range(num_layers):
            layer = _DenseLayer(in_channels + i * growth_rate, growth_rate, bottleneck_size, dropout)
            self.add_module(f'denselayer{i + 1}', layer)

    def forward(self, x):
        xs = [x]

        for name, layer in self.items():
            x_new = layer(xs)
            xs.append(x_new)

        return torch.cat(xs, 1)


class DenseNet(nn.Module):
    def __init__(self, block_config, growth_rate=16, bottleneck_size=4, num_classes=10, dropout=0.0):
        super(DenseNet, self).__init__()

        in_channels = 2 * growth_rate
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, in_channels, 7, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(in_channels)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))

        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, in_channels, growth_rate, bottleneck_size, dropout)
            self.features.add_module(f'denseblock{i + 1}', block)
            in_channels += num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = _Transition(in_channels, in_channels // 2)
                self.features.add_module(f'transition{i + 1}', trans)
                in_channels = in_channels // 2

        self.features.add_module(f'norm{i + 2}', nn.BatchNorm2d(in_channels))
        self.features.add_module(f'relu{i + 2}', nn.ReLU(inplace=True))

        self.classifier = nn.Linear(in_channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(self.features(x), (1, 1))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
