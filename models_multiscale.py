# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 16:24:22 2024

@author: usouu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleCNN_1(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(MultiScaleCNN_1, self).__init__()

        # 分支1：小尺度特征
        self.branch1_conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.branch1_bn1 = nn.BatchNorm2d(32)
        self.branch1_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.branch1_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.branch1_bn2 = nn.BatchNorm2d(64)
        self.branch1_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 分支2：中尺度特征
        self.branch2_conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2)
        self.branch2_bn1 = nn.BatchNorm2d(32)
        self.branch2_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.branch2_conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.branch2_bn2 = nn.BatchNorm2d(64)
        self.branch2_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 分支3：大尺度特征
        self.branch3_conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=1, padding=3)
        self.branch3_bn1 = nn.BatchNorm2d(32)
        self.branch3_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.branch3_conv2 = nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3)
        self.branch3_bn2 = nn.BatchNorm2d(64)
        self.branch3_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 融合特征
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # 分支1
        branch1 = F.relu(self.branch1_bn1(self.branch1_conv1(x)))
        branch1 = self.branch1_pool1(branch1)
        branch1 = F.relu(self.branch1_bn2(self.branch1_conv2(branch1)))
        branch1 = self.branch1_pool2(branch1)

        # 分支2
        branch2 = F.relu(self.branch2_bn1(self.branch2_conv1(x)))
        branch2 = self.branch2_pool1(branch2)
        branch2 = F.relu(self.branch2_bn2(self.branch2_conv2(branch2)))
        branch2 = self.branch2_pool2(branch2)

        # 分支3
        branch3 = F.relu(self.branch3_bn1(self.branch3_conv1(x)))
        branch3 = self.branch3_pool1(branch3)
        branch3 = F.relu(self.branch3_bn2(self.branch3_conv2(branch3)))
        branch3 = self.branch3_pool2(branch3)

        # 融合分支
        branch1 = self.global_pool(branch1).view(x.size(0), -1)
        branch2 = self.global_pool(branch2).view(x.size(0), -1)
        branch3 = self.global_pool(branch3).view(x.size(0), -1)

        # 拼接特征
        combined = torch.cat([branch1, branch2, branch3], dim=1)

        # 全连接层
        x = F.relu(self.fc1(combined))
        x = self.fc2(x)

        return x
