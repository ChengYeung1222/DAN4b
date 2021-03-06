#!/usr/bin/env Python
# coding=utf-8

from __future__ import print_function
from datetime import datetime
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import math
import data_loader
from custom_data_io import custom_dset
from torch.utils.data import DataLoader
import Models as models
from torch.utils import model_zoo

import radam

from sklearn import metrics

import visdom
import numpy as np

import logging
import time

np.seterr(divide='ignore', invalid='ignore')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG,format='%(asctime)s-%(levelname)s-%(message)s')
# logging.disable(level=logging.CRITICAL)

rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
# print(os.getcwd())
log_path = os.getcwd() + '/Logs/'
if not os.path.isdir(log_path):
    os.mkdir(log_path)
log_name = log_path + rq + '.log'
logfile = log_name
fh = logging.FileHandler(logfile, mode='w')
fh.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)

logger.addHandler(fh)

# To use this:
# python -m visdom.server
# http://localhost:8097/
vis = visdom.Visdom(env=u'ssd_DAN_Alex_1500')  # todo

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
batch_size = 256  # todo
epochs = 50  # depth: 1500 epoch: 48  auc: 0.928#todo
lr = 1e-3  # todo:1e-4,1e-3
momentum = 0.9
no_cuda = False
seed = 5  # todo:5,38;8,50;1, 26;2,2
# todo: dygz,32:1;ssd: 30, 11
log_interval = 30  # 20
log_interval_test = 11  # 30
l2_decay = 1e-3  # todo:5e-4,1e-3,5e-3
root_path = "./"
source_list = "./ssd/ssd_shallow_s.csv"
target_list = "./ssd/ssd_deep_s.csv"  # todo: 70500
validation_list = './ssd/ssd_deep_s.csv'
source_name = 'shallow zone'  # todo
target_name = 'deep zone'
test_name = 'deep zone/validation'
ckpt_path = './ckpt_d1500_ssd/'  # todo:wommd
ckpt_model = './ckpt/model_epoch1.pth'

# Create parent path if it doesn't exist
if not os.path.isdir(ckpt_path):
    os.mkdir(ckpt_path)

cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}  # torch.utils.data.DataLoader()

source_dataset = custom_dset(txt_path=source_list, nx=227, nz=227, labeled=True)
target_dataset = custom_dset(txt_path=target_list, nx=227, nz=227, labeled=True)  # todo:transform=None
validation_dataset = custom_dset(txt_path=validation_list, nx=227, nz=227, labeled=True)

source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True,
                           drop_last=True)
target_train_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True,
                                 drop_last=True)
target_test_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=10,
                                pin_memory=True, drop_last=True)

# source_loader = data_loader.load_training(root_path, source_name, batch_size, kwargs)
# target_train_loader = data_loader.load_training(root_path, target_name, batch_size, kwargs)
# target_test_loader = data_loader.load_testing(root_path, target_name, batch_size, kwargs)

len_source_dataset = len(source_dataset)  # a2817
len_target_dataset = len(target_dataset)
len_test_dataset = len(validation_dataset)
len_source_loader = len(source_loader)  # a88
len_target_loader = len(target_train_loader)  # w24
len_test_loader = len(target_test_loader)

with open(source_list, 'r') as f:
    lines = f.readlines()
    len_source_ones = 0
    len_source_zeros = 0
    for line in lines:
        items = line.split(',')

        if int(items[1]) == 1:
            len_source_ones += 1
    len_source_zeros = len_source_dataset - len_source_ones

with open(validation_list, 'r') as f:
    lines = f.readlines()
    len_val_ones = 0
    len_val_zeros = 0
    for line in lines:
        items = line.split(',')

        if int(items[1]) == 1:
            len_val_ones += 1
    len_val_zeros = len_test_dataset - len_val_ones


def load_pretrain_alex(model, alexnet_model=True):
    url = 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
    if alexnet_model == True:

        pretrained_dict = model_zoo.load_url(url)
        model_dict = model.state_dict()
        for k, v in pretrained_dict.items():
            if not 'classifier' in k:
                if not "features.0" in k:
                    if not "cls" in k:
                        model_dict[k] = pretrained_dict[k]
    return model


def load_pretrain(model, resnet_model=True):
    url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
    if resnet_model == True:
        pretrained_dict = model_zoo.load_url(url)
        model_dict = model.state_dict()
        for k, v in model_dict.items():
            if not "cls_fc" in k:
                # model_dict[k] = pretrained_dict[k[k.find(".") + 1:]]
                if not 'sharedNet.conv1.weight' in k:
                    if not 'sharedNet.bn1.weight' in k:
                        if not 'sharedNet.bn1.bias' in k:
                            if not 'sharedNet.bn1.running_mean' in k:
                                if not 'sharedNet.bn1.running_mean' in k:
                                    model_dict[k] = pretrained_dict[k[k.find(".") + 1:]]
    return model


def train(epoch, model, heterogeneity, optimizer_arg='radam', blending=True):
    LEARNING_RATE = lr / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)  # todo:denominator: epochs
    print('learning rate{: .6f}'.format(LEARNING_RATE))
    # ResNet optimizer
    # optimizer = torch.optim.Adam([
    #     {'params': model.sharedNet.parameters()},  # lr=LEARNING_RATE / 10
    #     {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
    # ], lr=LEARNING_RATE / 10, weight_decay=l2_decay)
    # AlexNet optimizeri;'k
    if optimizer_arg == 'Adam':
        optimizer = torch.optim.Adam(  # filter(lambda p: p.requires_grad,
            [  # {'params': model.conv1.parameters(), 'lr': LEARNING_RATE},
                {'params': filter(lambda p: p.requires_grad, model.features.parameters())},  # lr=LEARNING_RATE / 10
                {'params': model.l6.parameters(), 'lr': LEARNING_RATE},
                {'params': model.cls1.parameters(), 'lr': LEARNING_RATE},
                {'params': model.cls2.parameters(), 'lr': LEARNING_RATE},
                {'params': model.l7.parameters(), 'lr': LEARNING_RATE},
                {'params': model.cls4.parameters(), 'lr': LEARNING_RATE},
                {'params': model.l8.parameters(), 'lr': LEARNING_RATE},
                {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
            ], lr=LEARNING_RATE / 10, weight_decay=l2_decay)  # todo:momentum=momentum,

    elif optimizer_arg == 'radam':
        optimizer = radam.RAdam(params=[  # {'params': model.conv1.parameters(), 'lr': LEARNING_RATE},
            {'params': filter(lambda p: p.requires_grad, model.features.parameters())},  # lr=LEARNING_RATE / 10
            {'params': model.l6.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls2.parameters(), 'lr': LEARNING_RATE},
            {'params': model.l7.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls4.parameters(), 'lr': LEARNING_RATE},
            {'params': model.l8.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, weight_decay=l2_decay)

    elif optimizer_arg == 'adamw':
        optimizer = radam.AdamW(params=[  # {'params': model.conv1.parameters(), 'lr': LEARNING_RATE},
            {'params': filter(lambda p: p.requires_grad, model.features.parameters())},  # lr=LEARNING_RATE / 10
            {'params': model.l6.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls2.parameters(), 'lr': LEARNING_RATE},
            {'params': model.l7.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls4.parameters(), 'lr': LEARNING_RATE},
            {'params': model.l8.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, weight_decay=l2_decay)

    model.train()

    iter_source = iter(source_loader)
    iter_target = iter(target_train_loader)
    num_iter = max(len_source_loader, len_target_loader)  # 88
    TP, TN, FN, FP = 0, 0, 0, 0
    for i in range(1, num_iter):
        if i == 2:
            print(i)
        logging.debug('Start %dth iteration...' % (i))
        print('Start %dth iteration...' % (i))
        logging.debug('data_source, label_source = next(iter_source)')
        data_source, label_source, coordinate_source, fluid_source = next(iter_source)
        logging.debug('data_target, _ = next(iter_target)')
        data_target, _, coordinate_target, fluid_target = next(iter_target)
        logging.debug('if i % len_target_loader == 0:')
        if len_source_dataset < len_target_dataset:
            if i % len_source_loader == 0:
                iter_source = iter(source_loader)
        else:
            if i % len_target_loader == 0:
                logging.debug('iter_target = iter(target_train_loader)')
                iter_target = iter(target_train_loader)

        logging.debug('if cuda:')
        if cuda:
            logging.debug('push source data to gpu')
            data_source, label_source, coordinate_source, fluid_source = data_source.cuda(), label_source.cuda(), coordinate_source.cuda(), fluid_source.cuda()
            logging.debug('push target data to gpu')
            data_target, coordinate_target, fluid_target = data_target.cuda(), coordinate_target.cuda(), fluid_target.cuda()

        logging.debug('Variable source data')
        data_source, label_source, coordinate_source, fluid_source = Variable(data_source.float()), Variable(
            label_source.long()), Variable(coordinate_source.float()), Variable(fluid_source.float())
        logging.debug('Variable target data')
        data_target, coordinate_target, fluid_target = Variable(data_target.float()), Variable(
            coordinate_target.float()), Variable(fluid_target.float())

        logging.debug('clear old gradients from the last step')
        optimizer.zero_grad()

        score_source_pred, loss_mmd, new_feature_pred = model(data_source, data_target, coordinate_source,
                                                              coordinate_target, fluid_source, fluid_target,
                                                              heterogeneity=heterogeneity, blending=blending)  # todo
        logging.debug('Calculating loss_cls...')

        # softmax_score = F.softmax(score_source_pred, dim=1)  # todo: parallel
        if blending == True:
            # print(new_feature_pred)
            prob_source_pred = F.softmax(new_feature_pred, dim=1)
            y_1 = prob_source_pred[:, 1]
            y_0 = prob_source_pred[:, 0]
            ratio_source = y_1 / y_0 * len_source_zeros / len_source_ones
            prob_source_pred_new = torch.Tensor(prob_source_pred.size()).cuda()
            prob_source_pred_new[:, 1] = ratio_source.data / (ratio_source.data + 1)
            prob_source_pred_new[:, 0] = 1 / (ratio_source.data + 1)
            prob_source_pred_new = Variable(prob_source_pred_new)
            # softmax_new_feature = F.softmax(new_feature_pred, dim=1).cuda()
            # blending_softmax = 0.6 * softmax_score + 0.4 * softmax_new_feature
            # blending_log_softmax = torch.log(blending_softmax)
            # loss_cls = F.nll_loss(F.log_softmax(blending_log_softmax, dim=1),
            #                       target=label_source)
            blending_softmax = F.log_softmax(new_feature_pred, dim=1)
            # print('blending_softmax:')
            # print(blending_softmax)
            loss_cls = F.nll_loss(F.log_softmax(new_feature_pred, dim=1),
                                  target=label_source)
        else:
            prob_source_pred = F.softmax(score_source_pred, dim=1)
            y_1 = prob_source_pred[:, 1]
            y_0 = prob_source_pred[:, 0]
            ratio_source = y_1 / y_0 * len_source_zeros / len_source_ones
            prob_source_pred_new = torch.Tensor(prob_source_pred.size()).cuda()
            prob_source_pred_new[:, 1] = ratio_source.data / (ratio_source.data + 1)
            prob_source_pred_new[:, 0] = 1 / (ratio_source.data + 1)
            prob_source_pred_new = Variable(prob_source_pred_new)
            loss_cls = F.nll_loss(F.log_softmax(score_source_pred, dim=1),
                                  target=label_source)  # the negative log likelihood loss
            logging.debug('loss_cls = %s' % (loss_cls))
        gamma = 2 / (1 + math.exp(-10 * (epoch) / epochs)) - 1  # lambda in DAN paper#todo:denominator: epochs
        logging.debug('Push mmd to gpu...')
        # !!!!

        # loss_mmd = loss_mmd.cuda()#todo:
        loss = loss_cls + gamma * loss_mmd
        logging.debug('Calculate total loss')

        # loss = loss_cls

        pred = prob_source_pred_new.data.max(1)[1]

        logging.debug('compute training auc, f1 score and accuracy')
        # prob=[]
        # index_max=prob_source_pred_new.data.max(1)[1]
        # for ii in range(batch_size):
        #     prob.append(prob_source_pred_new.data[ii,index_max[ii]])
        fpr, tpr, thresholds = metrics.roc_curve(y_true=label_source.data, y_score=prob_source_pred.data[:, 1],
                                                 pos_label=1)
        auc_value = metrics.auc(fpr, tpr)
        TP += ((pred == 1) & (label_source.data.view_as(pred) == 1)).cpu().sum()
        TN += ((pred == 0) & (label_source.data.view_as(pred) == 0)).cpu().sum()
        FN += ((pred == 0) & (label_source.data.view_as(pred) == 1)).cpu().sum()
        FP += ((pred == 1) & (label_source.data.view_as(pred) == 0)).cpu().sum()
        if (TP + FP) != 0:
            p = TP / (TP + FP)
        else:
            p = 0
        if (TP + FN) != 0:
            r = TP / (TP + FN)
        else:
            r = 0
        if (r + p) != 0:
            F1score = 2 * r * p / (r + p)
        else:
            F1score = 0

        train_acc = (TP + TN) / (TP + TN + FP + FN)

        logging.debug('computing the derivative of the loss w.r.t. the params')
        loss.backward()

        logging.debug('updating params based on the gradients')
        optimizer.step()

        if i % log_interval == 0:
            # opts = dict(xlabel='minibatches',
            #             ylabel='Loss',
            #             title='Training Loss',
            #             legend=['Loss']))
            vis.line(X=np.array([i + (epoch - 1) * len_source_loader]), Y=loss_cls.cpu().data.numpy(), win='loss_cls',
                     update='append',
                     opts={'title': 'CNN risk'})
            vis.line(X=np.array([i + (epoch - 1) * len_source_loader]), Y=np.array([gamma]), win='gamma',
                     update='append',
                     opts={'title': 'penalty parameter'})
            # !!!!!!!
            # vis.line(X=np.array([i + (epoch - 1) * len_source_loader]), Y=loss_mmd.cpu().data.numpy(),
            #          win='loss_mmd', update='append',
            #         opts={'title': 'loss of MK_MMD'})
            # vis.line(X=np.array([i + (epoch - 1) * len_source_loader]), Y=loss.cpu().data.numpy(), win='loss',
            #          update='append',
            #          opts={'title': 'total loss'})  # todo:mmd visualization
            vis.line(X=np.array([i + (epoch - 1) * len_source_loader]), Y=np.array([auc_value]),
                     win='training auc',
                     update='append',
                     opts={'title': 'training auc'})
            vis.line(X=np.array([i + (epoch - 1) * len_source_loader]), Y=np.array([F1score]),
                     win='training F1 score',
                     update='append',
                     opts={'title': 'training F1 score'})
            vis.line(X=np.array([i + (epoch - 1) * len_source_loader]), Y=np.array([train_acc]),
                     win='train_acc',
                     update='append',
                     opts={'title': 'training accuracy'})
            continue
            # todo:mmd result
            print('{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}'.format(
                datetime.now(), epoch, i * len(data_source), len_source_dataset,
                                       100. * i / len_source_loader, loss.data[0], loss_cls.data[0],
                # !!!!
                # loss_mmd))
                loss_mmd.data[0]))


def test(epoch, model, heterogeneity, blending):
    model.eval()

    test_loss = 0
    correct = 0
    TP, TN, FN, FP = 0, 0, 0, 0
    F1score = 0

    iter_test = iter(target_test_loader)
    num_iter_test = len_test_loader

    for i in range(1, num_iter_test):
        data, label, coordinate_target, fluid_target = next(iter_test)  # data_shape: torch.Size([32, 3, 224, 224])
        data, label, fluid_target = data.float(), label.long(), fluid_target.float()
        if cuda:
            data, label, fluid_target = data.cuda(), label.cuda(), fluid_target.cuda()
        data, label, fluid_target = Variable(data, volatile=True), Variable(label), Variable(fluid_target)
        s_output, _, new_feature_pred = model(data, data, coordinate_target, coordinate_target, fluid_target,
                                              fluid_target,
                                              heterogeneity=heterogeneity,
                                              blending=blending)

        prob_val_pred = F.softmax(s_output, dim=1)
        y_1 = prob_val_pred[:, 1]
        y_0 = prob_val_pred[:, 0]
        ratio_val = y_1 / y_0 * len_val_zeros / len_val_ones
        prob_val_pred_new = torch.Tensor(prob_val_pred.size()).cuda()
        prob_val_pred_new[:, 1] = ratio_val.data / (ratio_val.data + 1)
        prob_val_pred_new[:, 0] = 1 / (ratio_val.data + 1)
        prob_val_pred_new = Variable(prob_val_pred_new)
        if blending == True:
            test_loss += F.nll_loss(F.log_softmax(new_feature_pred, dim=1), label, size_average=False).data[
                0]  # sum up batch loss
            loss_val = F.nll_loss(F.log_softmax(new_feature_pred, dim=1), label).data[0]

        else:
            test_loss += F.nll_loss(F.log_softmax(s_output, dim=1), label, size_average=False).data[
                0]  # sum up batch loss
            loss_val = F.nll_loss(F.log_softmax(s_output, dim=1), label).data[0]
        pred = prob_val_pred_new.data.max(1)[
            1]  # get the index of the max log-probability, s_output_shape: torch.Size([32, 31])
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()

        # fpr, tpr, thresholds = metrics.roc_curve(y_true=label_source.data, y_score=prob_source_pred.data[:, 1],
        #                                          pos_label=1)
        # auc_value = metrics.auc(fpr, tpr)

        fpr, tpr, thresholds = metrics.roc_curve(y_true=label.data, y_score=prob_val_pred.data[:, 1],
                                                 pos_label=1)

        # if np.isnan(tpr).all():
        #     tpr=np.full(shape=tpr.shape,fill_value=1e-8)

        auc_value_test = metrics.auc(fpr, tpr)  # todo
        TP += ((pred == 1) & (label.data.view_as(pred) == 1)).cpu().sum()
        TN += ((pred == 0) & (label.data.view_as(pred) == 0)).cpu().sum()
        FN += ((pred == 0) & (label.data.view_as(pred) == 1)).cpu().sum()
        FP += ((pred == 1) & (label.data.view_as(pred) == 0)).cpu().sum()

        if (TP + FP) != 0:
            p = TP / (TP + FP)
        else:
            p = 0
        if (TP + FN) != 0:
            r = TP / (TP + FN)
        else:
            r = 0
        if (r + p) != 0:
            F1score = 2 * r * p / (r + p)
        else:
            F1score = 0
        if i % log_interval_test == 0:
            vis.line(X=np.array([i + (epoch - 1) * len_test_loader]), Y=np.array([auc_value_test]),
                     win='testing auc',
                     update='append',
                     opts={'title': 'testing auc'})
            vis.line(X=np.array([i + (epoch - 1) * len_test_loader]), Y=np.array([F1score]),
                     win='testing F1 score',
                     update='append',
                     opts={'title': 'testing F1 score'})
            vis.line(X=np.array([i + (epoch - 1) * len_test_loader]), Y=np.array([correct / len_test_dataset]),
                     win='test_acc',
                     update='append',
                     opts={'title': 'testing accuracy'})
            vis.line(X=np.array([i + (epoch - 1) * len_test_loader]), Y=np.array([loss_val]),
                     win='loss_val',
                     update='append',
                     opts={'title': 'testing loss'})

    test_loss /= len_test_dataset
    print('\n{}  {} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), F1score: {}, auc: {}\n'.format(
        datetime.now(), test_name, test_loss, correct, len_test_dataset,
        100. * correct / len_test_dataset, F1score, auc_value_test))

    return correct


if __name__ == '__main__':
    # model = models.DANNet(num_classes=2)  # Models.py#todo:ResNet
    model = models.DAN_with_Alex(num_classes=2)
    correct = 0
    print(model)
    if cuda:
        model.cuda()
    model = load_pretrain_alex(model, alexnet_model=True)
    for epoch in range(1, epochs + 1):
        train(epoch, model, heterogeneity=False, blending=True)  # TODO
        t_correct = test(epoch, model, heterogeneity=False, blending=True)
        # Save models.
        ckpt_name = os.path.join(ckpt_path, 'model_epoch' + str(epoch) + '.pth')
        print('Save model: {}'.format(ckpt_name))
        torch.save(obj=model.state_dict(), f=ckpt_name)
        if t_correct > correct:
            correct = t_correct
        current_acc = 100. * t_correct / len_test_dataset
        print('{} source: {} to target: {} max correct: {} max accuracy{: .2f}%\n'.format(
            datetime.now(), source_name, target_name, correct, 100. * correct / len_test_dataset))
