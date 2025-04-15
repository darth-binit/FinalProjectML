###############################
#--- Second model

import torch.nn as nn
import torch
class ChaBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8, kernel_size=7):
        super(ChaBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x)
        x_channel = x * ca

        avg_out = torch.mean(x_channel, dim=1, keepdim=True)
        max_out, _ = torch.max(x_channel, dim=1, keepdim=True)
        x_spatial = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(x_spatial)

        return x_channel * sa

class Mha2D(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super(Mha2D, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        self.d_k = in_channels // num_heads

        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        query = self.query_conv(x).view(B, self.num_heads, self.d_k, H * W)
        key = self.key_conv(x).view(B, self.num_heads, self.d_k, H * W)
        value = self.value_conv(x).view(B, self.num_heads, self.d_k, H * W)

        query = query.permute(0, 1, 3, 2)
        scores = torch.matmul(query, key) / (self.d_k ** 0.5)
        attn = self.softmax(scores)
        out = torch.matmul(attn, value.permute(0, 1, 3, 2))
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        return x + out


class Bb(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, use_cbam=True, use_mha=True, num_heads=4):
        super(Bb, self).__init__()
        self.use_cbam = use_cbam
        self.use_mha = use_mha

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

        if self.use_cbam:
            self.cbam = ChaBAM(planes)
        if self.use_mha:
            self.mha = Mha2D(planes, num_heads)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        if self.use_cbam:
            out = self.cbam(out)
        if self.use_mha:
            out = self.mha(out)
        return out


class ResNetAttn(nn.Module):
    def __init__(self, block, layers, num_classes=7, num_heads=4):
        super(ResNetAttn, self).__init__()
        self.in_planes = 64
        self.num_heads = num_heads

        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Apply attention only to layer3 and layer4
        self.layer1 = self._make_layer(block, 64, layers[0], use_cbam=False, use_mha=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, use_cbam=False, use_mha=False)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, use_cbam=True, use_mha=True)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, use_cbam=True, use_mha=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def _make_layer(self, block, planes, blocks, stride=1, use_cbam=False, use_mha=False):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [
            block(self.in_planes, planes, stride, downsample,
                  use_cbam=use_cbam, use_mha=use_mha, num_heads=self.num_heads)
        ]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes,
                                use_cbam=use_cbam, use_mha=use_mha, num_heads=self.num_heads))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)