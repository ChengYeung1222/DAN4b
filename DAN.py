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
import ResNet as models
from torch.utils import model_zoo

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
batch_size = 32
epochs = 200
lr = 0.01
momentum = 0.9
no_cuda = False
seed = 8
log_interval = 10
l2_decay = 5e-4
root_path = "./"
source_list = "./Cut_off_grade_1_train_list.csv"
target_list = "./Cut_off_grade_1_val_list.csv"
source_name = 'source_name'  # todo
target_name = 'target_name'
ckpt_path = './ckpt/'
ckpt_model = ''

# Create parent path if it doesn't exist
if not os.path.isdir(ckpt_path):
    os.mkdir(ckpt_path)

cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}  # torch.utils.data.DataLoader()

source_dataset = custom_dset(txt_path=source_list, nx=227, nz=227)  # todo:
target_dataset = custom_dset(txt_path=target_list, nx=227, nz=227)

source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True,
                           drop_last=True)
target_train_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True,
                                 drop_last=True)
target_test_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)

# source_loader = data_loader.load_training(root_path, source_name, batch_size, kwargs)
# target_train_loader = data_loader.load_training(root_path, target_name, batch_size, kwargs)
# target_test_loader = data_loader.load_testing(root_path, target_name, batch_size, kwargs)

len_source_dataset = len(source_dataset)  # a2817
len_target_dataset = len(target_dataset)  # w795
len_source_loader = len(source_loader)  # a88
len_target_loader = len(target_train_loader)  # w24


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
    print('learning rate{: .4f}'.format(LEARNING_RATE))
    optimizer = torch.optim.SGD([
        {'params': model.sharedNet.parameters()},  # lr=LEARNING_RATE / 10
        {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
    ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)

    model.train()

    iter_source = iter(source_loader)
    iter_target = iter(target_train_loader)
    num_iter = len_source_loader  # 88
    for i in range(1, num_iter):
        data_source, label_source = next(iter_source)
        data_target, _ = next(iter_target)
        if i % len_target_loader == 0:  # i % 24
            iter_target = iter(target_train_loader)
        if cuda:
            data_source, label_source = data_source.cuda(), label_source.cuda()
            data_target = data_target.cuda()
        data_source, label_source = Variable(data_source.float()), Variable(label_source.long())
        data_target = Variable(data_target.float())

        optimizer.zero_grad()
        score_source_pred, loss_mmd = model(data_source, data_target)
        loss_cls = F.nll_loss(F.log_softmax(score_source_pred, dim=1),
                              target=label_source)  # the negative log likelihood loss
        gamma = 2 / (1 + math.exp(-10 * (epoch) / epochs)) - 1  # lambda in DAN paper
        loss = loss_cls + gamma * loss_mmd
        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print('{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}'.format(
                datetime.now(), epoch, i * len(data_source), len_source_dataset,
                                       100. * i / len_source_loader, loss.data[0], loss_cls.data[0], loss_mmd.data[0]))


def test(model):
    model.eval()
    test_loss = 0
    correct = 0

    for data, label in target_test_loader:  # data_shape: torch.Size([32, 3, 224, 224])
        data, label = data.float(), label.long()
        if cuda:
            data, label = data.cuda(), label.cuda()
        data, label = Variable(data, volatile=True), Variable(label)
        s_output, t_output = model(data, data)
        test_loss += F.nll_loss(F.log_softmax(s_output, dim=1), label, size_average=False).data[0]  # sum up batch loss
        pred = s_output.data.max(1)[1]  # get the index of the max log-probability, s_output_shape: torch.Size([32, 31])
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()

    test_loss /= len_target_dataset
    print('\n{}  {} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        datetime.now(), target_name, test_loss, correct, len_target_dataset,
        100. * correct / len_target_dataset))

    return correct


if __name__ == '__main__':
    model = models.DANNet(num_classes=2)  # ResNet.py#todo:
    correct = 0
    print(model)
    if cuda:
        model.cuda()
    model = load_pretrain(model)
    for epoch in range(1, epochs + 1):
        train(epoch, model)
        t_correct = test(model)
        ckpt_name = os.path.join(ckpt_path, 'model_epoch' + str(epoch) + '.pth')
        torch.save(obj=model.state_dict(), f=ckpt_name)
        if t_correct > correct:
            correct = t_correct
        print('{} source: {} to target: {} max correct: {} max accuracy{: .2f}%\n'.format(
            datetime.now(), source_name, target_name, correct, 100. * correct / len_target_dataset))
