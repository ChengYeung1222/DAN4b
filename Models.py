import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import mmd
import torch

# from torchvision import models

__all__ = ['ResNet', 'resnet50']
# __all__ = ['AlexNet', 'alexnet']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):  # block=Bottleneck,layers=[3,4,6,3]
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
                               bias=False, )  # todo:input_channels
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(
            inplace=True)  # inplace means that it will not allocate new memory and change tensors inplace.
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.baselayer = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
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
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = x.unsqueeze(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # before classifier

        return x


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),  # todo
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class AlexNetFc(nn.Module):
    def __init__(self):
        super(AlexNetFc, self).__init__()
        model_alexnet = AlexNet()
        self.features = model_alexnet.features
        # self.x_view = 0  # todo:
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i), model_alexnet.classifier[i])
        self.__in_features = model_alexnet.classifier[6].in_features  # nn.Linear(in_features=,)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        # self.x_view = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self.__in_features


# self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(256 * 6 * 6, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, num_classes),
#         )
class DAN_with_Alex(nn.Module):

    def __init__(self, num_classes=2):
        super(DAN_with_Alex, self).__init__()
        # self.conv1=alexnet().features[0]
        self.features = alexnet().features
        # for i in range(1,13):
        #     exec('self.features{} = alexnet().features[{}]'.format(i, i))

        self.l6 = alexnet().classifier[0]
        self.cls1 = alexnet().classifier[1]
        self.cls2 = alexnet().classifier[2]
        self.l7 = alexnet().classifier[3]
        self.cls4 = alexnet().classifier[4]
        self.l8 = alexnet().classifier[5]
        self.cls_fc = nn.Linear(4096, num_classes)

    def forward(self, source, target):
        loss = 0
        # source=self.conv1(source)
        source = self.features(source)
        source = source.view(source.size(0), 256 * 6 * 6)
        source = self.l6(source)
        if self.training == True:
            # target=self.conv1(target)
            target = self.features(target)
            target = target.view(target.size(0), 256 * 6 * 6)
            target = self.l6(target)
            # !!!!!!!!
            loss += mmd.mmd_rbf_noaccelerate(source, target)  # todo: add mmd
        self.cls1.cuda()
        self.cls2.cuda()
        source = self.cls1(source)
        source = self.cls2(source)
        if self.training == True:
            target = self.cls1(target)
            target = self.cls2(target)
        source = self.l7(source)
        if self.training == True:
            target = self.l7(target)
            # !!!!!!!!
            loss += mmd.mmd_rbf_noaccelerate(source, target)
        source = self.l7(source)
        self.cls4.cuda()
        source = self.cls4(source)
        if self.training == True:
            target = self.l7(target)
            target = self.cls4(target)
        source = self.l8(source)
        if self.training == True:
            target = self.l8(target)
            # !!!!!!!!
            loss += mmd.mmd_rbf_noaccelerate(source, target)
        source = self.cls_fc(source)
        # if self.training == True:
        #     target = alexnet().classifier[6](target)
        return source, loss


def alexnet(pretrained=False, **kwargs):
    model = AlexNet()
    for name, params in model.named_parameters():
        if name.find('conv') != -1:
            torch.nn.init.xavier_normal(params[0])
        elif name.find('fc') != -1:
            torch.nn.init.xavier_normal(params[0])
            torch.nn.init.xavier_normal(params[1])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']), strict=False)
    return model


class DANNet(nn.Module):

    def __init__(self, num_classes=2):  # todo
        super(DANNet, self).__init__()
        self.sharedNet = resnet50(pretrained=False)
        self.cls_fc = nn.Linear(8192, num_classes)  # todo:2048

    def forward(self, source, target):
        loss = 0
        source = self.sharedNet(source)
        if self.training == True:
            target = self.sharedNet(target)
            # loss += mmd.mmd_rbf_accelerate(source, target)
            loss += mmd.mmd_rbf_noaccelerate(source, target)

        source = self.cls_fc(source)  # todo:
        # target = self.cls_fc(target)

        return source, loss


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    for name, params in model.named_parameters():
        if name.find('conv') != -1:
            torch.nn.init.kaiming_normal_(params[0])
        elif name.find('fc') != -1:
            torch.nn.init.kaiming_normal_(params[0])
            torch.nn.init.kaiming_normal_(params[1])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model
