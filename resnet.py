'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

Modified based on PyTorch Implementation and
Tile2Vect: N. Jean, et al. “Tile2Vec: Unsupervised Representation Learning for Spatially Distributed Data,” AAAI 2019.
'''

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, h_dim = 512, in_channels = 3, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.h_dim = h_dim

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # following Tile2Vec, maxpool is not used here

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer5 = self._make_layer(block, h_dim, layers[3], stride=2) # same as Tile2Vec

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc  = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)

        return out


class ResNet_Encode(nn.Module):
    def __init__(self, block, layers, h_dim = 512, in_channels = 3):
        super(ResNet_Encode, self).__init__()
        self.in_planes = 64
        self.h_dim = h_dim

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # following Tile2Vec, maxpool is not used here

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer5 = self._make_layer(block, h_dim, layers[3], stride=2) # one additional layer same as Tile2Vec

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out) # one additional layer same as Tile2Vec

        out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)

        return out



def ResNet18(h_dim=512, in_channels = 3, num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2, 2], h_dim, in_channels, num_classes)

def ResNet18_Encode(h_dim=512, in_channels = 3):
    return ResNet_Encode(BasicBlock, [2, 2, 2, 2, 2], h_dim, in_channels)


def ResNet34(h_dim=512, in_channels = 3, num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3, 3], h_dim, in_channels, num_classes)

def ResNet34_Encode(h_dim=512, in_channels = 3):
    return ResNet_Encode(BasicBlock, [3, 4, 6, 3, 3], h_dim, in_channels)


def ResNet50(h_dim=512, in_channels = 3, num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3, 3], h_dim, in_channels, num_classes)

def ResNet50_Encode(h_dim=512, in_channels = 3):
    return ResNet_Encode(Bottleneck, [3, 4, 6, 3, 3], h_dim, in_channels)


####################################
# The following codes are for pre + post images bitemporal change detection or joint classification
####################################

class ResNet_Bi(nn.Module):
    def __init__(self, block, layers, h_dim = 512, in_channels = 3, num_classes=1000):
        super(ResNet_Bi, self).__init__()
        self.in_planes = 64
        self.h_dim = h_dim

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # following Tile2Vec, maxpool is not used here

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer5 = self._make_layer(block, h_dim, layers[3], stride=2) # same as Tile2Vec

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc  = nn.Linear(2*512*block.expansion, num_classes) # 2 is for pre + post features

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x1, x2):
        #########################
        # x1
        #########################
        out1 = self.conv1(x1)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        # out = self.maxpool(out)

        out1 = self.layer1(out1)
        out1 = self.layer2(out1)
        out1 = self.layer3(out1)
        out1 = self.layer4(out1)
        out1 = self.layer5(out1)

        out1 = self.avgpool(out1)
        out1 = torch.flatten(out1, start_dim=1)

        #########################
        # x2
        #########################
        out2 = self.conv1(x2)
        out2 = self.bn1(out2)
        out2 = self.relu(out2)
        # out = self.maxpool(out)

        out2 = self.layer1(out2)
        out2 = self.layer2(out2)
        out2 = self.layer3(out2)
        out2 = self.layer4(out2)
        out2 = self.layer5(out2)

        out2 = self.avgpool(out2)
        out2 = torch.flatten(out2, start_dim=1)

        #########################
        # feature fusion for change detection or joint classification
        #########################
        out = self.fc(torch.cat((out1, out2), dim=1))

        return out


def ResNet18_Bi(h_dim=512, in_channels = 3, num_classes=1000):
    return ResNet_Bi(BasicBlock, [2, 2, 2, 2, 2], h_dim, in_channels, num_classes)

def ResNet34_Bi(h_dim=512, in_channels = 3, num_classes=1000):
    return ResNet_Bi(BasicBlock, [3, 4, 6, 3, 3], h_dim, in_channels, num_classes)

def ResNet50_Bi(h_dim=512, in_channels = 3, num_classes=1000):
    return ResNet_Bi(Bottleneck, [3, 4, 6, 3, 3], h_dim, in_channels, num_classes)
