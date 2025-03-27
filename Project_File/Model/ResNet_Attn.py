import torch.nn as nn
import torch

class ChBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8, kernel_size=7):
        """
        CBAM (Convolutional Block Attention Module) applies both channel and spatial attention.
        """
        super(ChBAM, self).__init__()
        # Channel Attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Squeeze spatial dimensions
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Apply channel attention
        ca = self.channel_attention(x)
        x_channel = x * ca

        # Apply spatial attention
        avg_out = torch.mean(x_channel, dim=1, keepdim=True)
        max_out, _ = torch.max(x_channel, dim=1, keepdim=True)
        x_spatial = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(x_spatial)

        return x_channel * sa


# %%
# ----------------------------
# Define Multi-Head Self-Attention Module for 2D Feature Maps
# ----------------------------
class MultiHeadAttention2D(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        """
        A simple multi-head self-attention for 2D feature maps.
        Splits channels into heads, applies scaled dot-product attention, then reassembles.
        """
        super(MultiHeadAttention2D, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        self.d_k = in_channels // num_heads  # per-head channel dimension

        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()  # Input shape: (B, C, H, W)
        # Linear projections
        query = self.query_conv(x)  # (B, C, H, W)
        key = self.key_conv(x)  # (B, C, H, W)
        value = self.value_conv(x)  # (B, C, H, W)

        # Reshape into (B, num_heads, d_k, H*W)
        query = query.view(B, self.num_heads, self.d_k, H * W)
        key = key.view(B, self.num_heads, self.d_k, H * W)
        value = value.view(B, self.num_heads, self.d_k, H * W)

        # Transpose query: (B, num_heads, H*W, d_k)
        query = query.permute(0, 1, 3, 2)
        # Compute attention scores: (B, num_heads, H*W, H*W)
        scores = torch.matmul(query, key) / (self.d_k ** 0.5)
        attn = self.softmax(scores)

        # Apply attention to value
        out = torch.matmul(attn, value.permute(0, 1, 3, 2))  # (B, num_heads, H*W, d_k)
        # Reassemble heads: reshape to (B, C, H, W)
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        # Residual connection
        return x + out


# %%
class BasicBlock(nn.Module):
    expansion = 1  # For BasicBlock, output channels = planes * expansion

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNetAttention(nn.Module):
    def __init__(self, block, layers: list = [], num_classes: int = 7, use_cbam: bool = True,
                 use_multihead: bool = True, num_heads: int = 4):
        """
        A custom ResNet-like model built from scratch.
        It uses a BasicBlock structure and includes attention modules.

        Args:
            block: The block class (e.g., BasicBlock)
            layers: A list with the number of blocks in each layer (e.g., [2, 2, 2, 2] for ResNet-18)
            num_classes: Number of output classes
            use_cbam: Whether to include CBAM after layer3
            use_multihead: Whether to include multi-head self-attention after layer3
            num_heads: Number of heads for the multi-head attention module
        """
        super(ResNetAttention, self).__init__()
        self.in_planes = 64
        self.use_cbam = use_cbam
        self.use_multihead = use_multihead

        # Initial convolution and pooling (similar to ResNet)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Build residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Insert attention modules after layer3
        if self.use_cbam:
            self.cbam = ChBAM(in_channels=256 * block.expansion)
        if self.use_multihead:
            self.mha = MultiHeadAttention2D(in_channels=256 * block.expansion, num_heads=num_heads)

        # Final pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512 * block.expansion, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256 * block.expansion, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes))

        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        # Initialize weights using Kaiming He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial conv and pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)  # Expected shape: (B, 256, H, W)

        # Apply attention modules after layer3 if enabled
        if self.use_cbam:
            x = self.cbam(x)
        if self.use_multihead:
            x = self.mha(x)

        x = self.layer4(x)  ##Expected shape : (B, 512, H, W)

        # Global average pooling and feedforward classifier head
        x = self.avgpool(x)  # (B, 512, 1, 1)
        x = torch.flatten(x, 1)  # (B, 512)
        x = self.fc(x)  # (B, num_classes)
        return x