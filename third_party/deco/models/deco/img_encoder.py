import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride


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


class ResNetBackbone(nn.Module):
    """DECO 使用的轻量 ResNet backbone。

    Kuavo-DECO 阶段三需要同时支持 RGB backbone 和 1-channel depth backbone。
    因此这里把原先固定 3 通道的 ResNet34 拆成可配置输入通道与层数的实现。
    """

    def __init__(self, in_channels=3, layers=(3, 4, 6, 3)):
        super(ResNetBackbone, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2)


    def _make_layer(self, block, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor): 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class ResNet18(ResNetBackbone):
    def __init__(self, in_channels=3):
        super().__init__(in_channels=in_channels, layers=(2, 2, 2, 2))


class ResNet34(ResNetBackbone):
    def __init__(self, in_channels=3):
        super().__init__(in_channels=in_channels, layers=(3, 4, 6, 3))


def build_resnet_backbone(backbone_name="resnet34", in_channels=3):
    """按配置构建 DECO 视觉 backbone。

    支持 resnet34 默认容量，也保留 resnet18 作为低显存/低延迟备选。
    """
    backbone_name = backbone_name.lower()
    if backbone_name == "resnet18":
        return ResNet18(in_channels=in_channels)
    if backbone_name == "resnet34":
        return ResNet34(in_channels=in_channels)
    raise ValueError(f"Unsupported DECO visual backbone: {backbone_name}")
