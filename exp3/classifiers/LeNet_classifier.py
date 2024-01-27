import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, num_classes=10, dropout=0.0):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        self.relu3 = nn.ReLU()

        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.flatten(x)
        x = self.dropout1(x)  # 添加Dropout层
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout2(x)  # 添加Dropout层
        x = self.fc2(x)
        return x