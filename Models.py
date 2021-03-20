import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import mmd
import torch
import logging
from SpatialCrossMapLRN_temp import SpatialCrossMapLRN_temp
from torch.autograd import Variable

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


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


def CrossMapLRN(size, alpha, beta, k=1.0, gpuDevice=0):
    if SpatialCrossMapLRN_temp is not None:
        lrn = SpatialCrossMapLRN_temp(size, alpha, beta, k, gpuDevice=gpuDevice)
        n = Lambda(lambda x, lrn=lrn: Variable(lrn.forward(x.data).cuda(gpuDevice)) if x.data.is_cuda else Variable(
            lrn.forward(x.data)))
    else:
        n = nn.LocalResponseNorm(size, alpha, beta, k).cuda(gpuDevice)
    return n


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


class new_Net(nn.Module):
    def __init__(self, num_classes=2):
        super(new_Net, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(6, 50),  # todo
            nn.ReLU(inplace=True),

            nn.Linear(50, 50),
            nn.ReLU(inplace=True),
            # nn.Linear(8, num_classes),
            nn.Linear(50, 50),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class AlexNet(nn.Module):

    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=11, stride=4, padding=2),  # todo
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # CrossMapLRN(5, 0.0001, 0.75),#todo
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # CrossMapLRN(5, 0.0001, 0.75),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        logging.debug('self.training=%d' % self.training)
        if self.training == True:
            self.classifier = nn.Sequential(

                nn.Dropout(0.5),  # todo
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),

                nn.Dropout(0.5),  # todo:0.5,0.7
                nn.Linear(4096, 4096),  # todo:4096
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),  # todo:2048

                nn.Linear(4096, 50),  # todo:parallel
                nn.ReLU(inplace=True),
                nn.Linear(50, num_classes),

            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 50),
                nn.ReLU(inplace=True),
                nn.Linear(50, num_classes),
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
        # ++++++++++
        self.cls_fc = nn.Linear(4096, num_classes)  # todo:parallel

        # self.cls_fc = nn.Linear(2048, num_classes)

    def forward(self, source, target, coordinate_source, coordinate_target, fluid_source, fluid_target, heterogeneity,
                blending=False, parallel=False, fluid_feature=0.):
        # fd_kernel=False):
        loss = .0
        # if heterogeneity == True:
        #     kernel_dist = mmd.distance_kernel(coordinate_source, coordinate_target)

        # kernel_i = mmd.guassian_kernel_no_loop(
        #     source.data.view(source.shape[0], source.shape[1] * source.shape[2] * source.shape[3]),
        #     target.data.view(target.shape[0], target.shape[1] * target.shape[2] * target.shape[3]))
        # source=self.conv1(source)
        source = self.features(source)  # todo:nmdwsm
        source = source.view(source.size(0), 256 * 6 * 6)

        # target=self.conv1(target)
        if blending == True:
            # new_features = Variable(torch.rand(256, 8), requires_grad=True)  # todo: parallel
            new_features = fluid_source
            # new_net().cuda()
            new_features = new_net().cuda().classifier[0](new_features)  # todo:parallel
            new_features = new_net().cuda().classifier[1](new_features)
        else:
            new_features = 0.

        source = self.l6(source)
        if self.training == True:
            target = self.features(target)
            target = target.view(target.size(0), 256 * 6 * 6)
            target = self.l6(target)
            # !!!!!!!!
            # if loss == 0.:#todo
            #     pass
            # else:
            # if blending != True:
            # loss += mmd.mmd_rbf_noaccelerate(source, target, coordinate_source, coordinate_target, kernel_i=0,
            #                                  heterogeneity=heterogeneity)  # todo: add mmd
            logging.debug('kernel loss = %s' % (loss))
        self.cls1.cuda()
        self.cls2.cuda()
        source = self.cls1(source)
        source = self.cls2(source)
        if self.training == True:
            target = self.cls1(target)
            target = self.cls2(target)
        source = self.l7(source)
        if blending == True:
            new_features = new_net().cuda().classifier[2](new_features)  # todo:parallel
            new_features = new_net().cuda().classifier[3](new_features)
        if self.training == True:
            target = self.l7(target)
            # !!!!!!!!
            # if blending != True:
            # loss += mmd.mmd_rbf_noaccelerate(source, target, coordinate_source, coordinate_target, kernel_i=0,
            #                                  heterogeneity=heterogeneity)  # todo
            logging.debug('kernel loss = %s' % (loss))
        self.cls4.cuda()
        source = self.cls4(source)
        if self.training == True:
            target = self.l7(target)
            target = self.cls4(target)
        source = self.l8(source)
        if blending == True:
            source = alexnet().cuda().classifier[7](source)
            source = alexnet().cuda().classifier[8](source)

        # self.l8 = alexnet().classifier[5]  # 7,8,9

        if self.training == True:
            target = self.l8(target)  # relu
            if blending == True:
                target = alexnet().cuda().classifier[7](target)
                target = alexnet().cuda().classifier[8](target)
            # !!!!!!!!
            # if blending != True:
            # loss += mmd.mmd_rbf_noaccelerate(source, target, coordinate_source, coordinate_target, kernel_i=0,
            #                                  heterogeneity=heterogeneity)  # todo:wommd
            logging.debug('kernel loss = %s' % (loss))
        # +++++++++++
        # new_features = Variable(torch.rand(256, 2), requires_grad=True).cuda()  # todo: parallel
        # source = torch.cat((source, new_features), dim=1)
        # if blending == True:
        #     new_features = new_net().cuda().classifier[4](new_features)  # todo:parallel
        #     new_features = new_net().cuda().classifier[5](new_features)
        #     new_cat = torch.cat((source, new_features), dim=1)
        #     fc_out = nn.Linear(new_cat.size(1), self.cls_fc.out_features).cuda()
        #     new_cat = fc_out(new_cat)
        if parallel == True:
            source = alexnet().cuda().classifier[7](source)
            source = alexnet().cuda().classifier[8](source)
            new_cat = torch.cat((source, fluid_feature), dim=1)
            fc_out = nn.Linear(new_cat.size(1), self.cls_fc.out_features).cuda()
            new_cat = fc_out(new_cat)
        else:
            source = self.cls_fc(source)
            new_cat = 0.

        # if self.training == True:
        #     target = alexnet().classifier[6](target)

        return source, loss, new_cat


def new_net():
    model = new_Net()
    for name, params in model.named_parameters():
        if name.find('classifier') != -1:
            torch.nn.init.normal(params)
    return model


def alexnet(pretrained=False, frozen=False, **kwargs):
    model = AlexNet()
    for name, params in model.named_parameters():
        if frozen:  # todo:
            if name.find('3') != -1:
                params.requires_grad = False
            if name.find('6') != -1:
                params.requires_grad = False
        if name.find('bias') == -1:
            if name.find('features') != -1:
                torch.nn.init.kaiming_normal(params)
            # if name.find('conv2')!=-1:

            elif name.find('classifier') != -1:
                torch.nn.init.kaiming_normal(params)
        else:
            torch.nn.init.constant(params, val=0.01)
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
