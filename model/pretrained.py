import torch
import torch.nn as nn
import math
from torchvision import models

# Pretrained version
class Selayer(nn.Module):
    def __init__(self, inplanes):
        super(Selayer, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(inplanes, inplanes // 16, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(inplanes // 16, inplanes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sigmoid(out)

        return x * out


class BottleneckX_Origin(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cardinality, stride=1, downsample=None):
        super(BottleneckX_Origin, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)

        self.conv2 = nn.Conv2d(planes * 2, planes * 2, kernel_size=3, stride=stride,
                               padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 2)

        self.conv3 = nn.Conv2d(planes * 2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.selayer = Selayer(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.selayer(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SEResNeXt_Origin(nn.Module):
    def __init__(self, block, layers, input_channels=3, cardinality=32, num_classes=1000):
        super(SEResNeXt_Origin, self).__init__()
        self.cardinality = cardinality
        self.inplanes = 64
        self.input_channels = input_channels

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.classifier = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        x = self.classifier(x)

        return x


class SEResNeXt_Half(SEResNeXt_Origin):
    def remove_half(self):
        self.layer3 = None
        self.layer4 = None
        self.avgpool = None
        self.fc = None
        self.classifier = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        return x


def load_weight(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(pretrained_dict)

def se_resnext_half(dump_path, **kwargs):
    model = SEResNeXt_Half(BottleneckX_Origin, [3, 4, 6, 3], **kwargs)

    network_weight = torch.load(dump_path, map_location=torch.device("cpu"))['weight']
    load_weight(model, network_weight)
    model.remove_half()

    return model

class ConvNext_block(nn.Module):
    def __init__(self):
        super(ConvNext_block, self).__init__()

        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.ln = nn.LayerNorm([32, 32])
        self.conv2 = nn.Conv2d(64, 256, 1, 1)
        self.gelu = nn.GELU()
        self.conv3 = nn.Conv2d(256, 64, 1, 1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.ln(y)
        y = self.conv2(y)
        y = self.gelu(y)
        y = self.conv3(y)

        y += x

        return y

class Rgb2hsv(nn.Module):
    def __init__(self):
        super(Rgb2hsv, self).__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

        self.res_blocks = nn.Sequential(*[ConvNext_block() for _ in range(3)])

        self.dec1 = nn.Sequential(
            nn.Conv2d(64+64, 32*4, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.PixelShuffle(2),
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(32+32, 16*4, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.PixelShuffle(2),
        )

        self.dec3 = nn.Sequential(
            nn.Conv2d(16+16, 3*4, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.PixelShuffle(2),
        )

    def forward(self, rgb):

        enc1 = self.enc1(rgb)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)

        hsv = self.res_blocks(enc3)

        hsv = self.dec1(torch.cat([hsv, enc3], 1))
        hsv = self.dec2(torch.cat([hsv, enc2], 1))
        hsv = self.dec3(torch.cat([hsv, enc1], 1))

        return hsv


def pretrain_rgb2hsv():
    model = Rgb2hsv()
    model.load_state_dict(torch.load("rgb2hsv3.pth", map_location=torch.device("cpu")))

    for i in model.parameters():
        i.requires_grad = False

    return model


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        # self.to_relu_3_3 = nn.Sequential()
        # self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        # for x in range(9, 16):
        #     self.to_relu_3_3.add_module(str(x), features[x])
        # for x in range(16, 23):
        #     self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.to_relu_1_2(x)
        x = self.to_relu_2_2(x)
        return x