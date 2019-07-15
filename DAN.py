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

import visdom
import numpy as np

import logging
import time

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
vis = visdom.Visdom(env=u'DAN_Alex')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
batch_size = 256  # todo
epochs = 200
lr = 1e-4
momentum = 0.9
no_cuda = False
seed = 8
log_interval = 20
l2_decay = 1e-4  # todo
root_path = "./"
source_list = "./train_list.csv"
target_list = "./pre_list.csv"
validation_list = './val_list.csv'
source_name = 'known zone'  # todo
target_name = 'unknown zone'
test_name = 'validation'
ckpt_path = './ckpt/'
ckpt_model = './ckpt/model_epoch1.pth'

# Create parent path if it doesn't exist
if not os.path.isdir(ckpt_path):
    os.mkdir(ckpt_path)

cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}  # torch.utils.data.DataLoader()

source_dataset = custom_dset(txt_path=source_list, nx=227, nz=227, labeled=True)
target_dataset = custom_dset(txt_path=target_list, nx=227, nz=227, labeled=False)  # todo:transform=None
validation_dataset = custom_dset(txt_path=validation_list, nx=227, nz=227, labeled=True)

source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                           drop_last=True)
target_train_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                                 drop_last=True)
target_test_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=16,
                                pin_memory=True)

# source_loader = data_loader.load_training(root_path, source_name, batch_size, kwargs)
# target_train_loader = data_loader.load_training(root_path, target_name, batch_size, kwargs)
# target_test_loader = data_loader.load_testing(root_path, target_name, batch_size, kwargs)

len_source_dataset = len(source_dataset)  # a2817
len_target_dataset = len(target_dataset)
len_test_dataset = len(validation_dataset)
len_source_loader = len(source_loader)  # a88
len_target_loader = len(target_train_loader)  # w24
len_test_loader = len(target_test_loader)


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
    # else:
    #     model.load_state_dict(torch.load(ckpt_model))
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
    # else:#todo:load ckpt
    #     model.load_state_dict(torch.load(ckpt_model))
    #     model_dict = model.state_dict()
    #     for k, v in model_dict.items():
    #         if not "cls_fc" in k:
    #             model_dict[k] = pretrained_dict[k[k.find(".") + 1:]]

    return model


def load_pretrain_v0(model):
    for n, _ in model.named_parameters():
        print(n)
    url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
    pretrained_dict = model_zoo.load_url(url)
    model_dict = model.state_dict()
    # filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}

    # overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    for k, v in model_dict.items():
        if not "cls_fc" in k:
            model_dict[k] = pretrained_dict[k[k.find(".") + 1:]]
    model.load_state_dict(model_dict)
    return model


def train(epoch, model):
    LEARNING_RATE = lr / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)
    print('learning rate{: .6f}'.format(LEARNING_RATE))
    # optimizer = torch.optim.Adam([
    #     {'params': model.sharedNet.parameters()},  # lr=LEARNING_RATE / 10
    #     {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
    # ], lr=LEARNING_RATE / 10, weight_decay=l2_decay)
    ## AlexNet optimizer
    optimizer = torch.optim.Adam([
        # {'params': model.conv1.parameters(), 'lr': LEARNING_RATE},
        {'params': model.features.parameters()},  # lr=LEARNING_RATE / 10
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
    num_iter = len_source_loader  # 88
    TP, TN, FN, FP = 0, 0, 0, 0
    for i in range(1, num_iter):
        logging.debug('Start %dth iteration...' % (i))
        print('Start %dth iteration...' % (i))
        logging.debug('data_source, label_source = next(iter_source)')
        data_source, label_source = next(iter_source)
        logging.debug('data_target, _ = next(iter_target)')
        data_target = next(iter_target)
        logging.debug('if i % len_target_loader == 0:')
        # if i % len_source_loader == 0:
        #     iter_source = iter(source_loader)
        if i % len_target_loader == 0:
            # if True:
            logging.debug('iter_target = iter(target_train_loader)')
            iter_target = iter(target_train_loader)

        logging.debug('if cuda:')
        if cuda:
            logging.debug('push source data to gpu')
            data_source, label_source = data_source.cuda(), label_source.cuda()
            logging.debug('push target data to gpu')
            data_target = data_target.cuda()

        logging.debug('Variable source data')
        data_source, label_source = Variable(data_source.float()), Variable(label_source.long())
        logging.debug('Variable target data')
        data_target = Variable(data_target.float())

        logging.debug('clear old gradients from the last step')
        optimizer.zero_grad()

        score_source_pred, loss_mmd = model(data_source, data_target)
        logging.debug('Calculating loss_cls...')
        loss_cls = F.nll_loss(F.log_softmax(score_source_pred, dim=1),
                              target=label_source)  # the negative log likelihood loss
        gamma = 2 / (1 + math.exp(-10 * (epoch) / epochs)) - 1  # lambda in DAN paper#todo:-50
        logging.debug('Push mmd to gpu...')
        # !!!!
        loss_mmd = loss_mmd.cuda()
        loss = loss_cls + gamma * loss_mmd
        logging.debug('Calculate total loss')

        # loss = loss_cls

        pred = score_source_pred.data.max(1)[1]

        logging.debug('compute training f1 score and accuracy')
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


def test(epoch, model):
    model.eval()
    test_loss = 0
    correct = 0
    TP, TN, FN, FP = 0, 0, 0, 0
    F1score = 0

    iter_test = iter(target_test_loader)
    num_iter_test = len_test_loader

    for i in range(1, num_iter_test):
        data, label = next(iter_test)  # data_shape: torch.Size([32, 3, 224, 224])
        data, label = data.float(), label.long()
        if cuda:
            data, label = data.cuda(), label.cuda()
        data, label = Variable(data, volatile=True), Variable(label)
        s_output, _ = model(data, data)
        test_loss += F.nll_loss(F.log_softmax(s_output, dim=1), label, size_average=False).data[0]  # sum up batch loss
        pred = s_output.data.max(1)[1]  # get the index of the max log-probability, s_output_shape: torch.Size([32, 31])
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()

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
        if i % log_interval == 0:
            vis.line(X=np.array([i + (epoch - 1) * len_test_loader]), Y=np.array([F1score]),
                     win='testing F1 score',
                     update='append',
                     opts={'title': 'testing F1 score'})
            vis.line(X=np.array([i + (epoch - 1) * len_test_loader]), Y=np.array([correct / len_test_dataset]),
                     win='test_acc',
                     update='append',
                     opts={'title': 'testing accuracy'})

    test_loss /= len_test_dataset
    print('\n{}  {} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), F1score: {}\n'.format(
        datetime.now(), test_name, test_loss, correct, len_test_dataset,
        100. * correct / len_test_dataset, F1score))

    return correct


if __name__ == '__main__':
    # model = models.DANNet(num_classes=2)  # Models.py#todo:
    model = models.DAN_with_Alex(num_classes=2)
    correct = 0
    print(model)
    if cuda:
        model.cuda()
    model = load_pretrain_alex(model)
    for epoch in range(1, epochs + 1):
        train(epoch, model)
        t_correct = test(epoch, model)
        # Save models.
        ckpt_name = os.path.join(ckpt_path, 'model_epoch' + str(epoch) + '.pth')
        print('Save model: {}'.format(ckpt_name))
        torch.save(obj=model.state_dict(), f=ckpt_name)
        if t_correct > correct:
            correct = t_correct
        current_acc = 100. * t_correct / len_test_dataset
        # vis.line(X=np.array([epoch]), Y=np.array([current_acc]), win='accuracy', update='append',
        #          opts={'title': 'current accuracy'})
        print('{} source: {} to target: {} max correct: {} max accuracy{: .2f}%\n'.format(
            datetime.now(), source_name, target_name, correct, 100. * correct / len_test_dataset))
