#!/usr/bin/env python
# encoding: utf-8

import torch
from torch.autograd import Variable

import logging
# logging.disable(level=logging.CRITICAL)
# logging.basicConfig(level=logging.DEBUG,format='%(asctime)s-%(levelname)s-%(message)s')


# import numpy as np


# Consider linear time MMD with a linear kernel:
# K(f(x), f(y)) = f(x)^Tf(y)
# h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)
#             = [f(x_i) - f(y_i)]^T[f(x_j) - f(y_j)]
#
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def mmd_linear(f_of_X, f_of_Y):
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    # in resnet: [64,64,2048]
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)),
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


def guassian_kernel_two_loops(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    L2_distance = torch.zeros(int(total.size(0)), int(total.size(0))).cuda()
    L2_distance = Variable(L2_distance)
    for i in range(total.size()[0]):
        for j in range(total.size()[0]):
            square = (total[i] - total[j]) ** 2
            L2_distance[i, j] = square.sum()

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


def guassian_kernel_one_loop(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    L2_distance = torch.zeros(int(total.size(0)), int(total.size(0))).cuda()
    L2_distance = Variable(L2_distance)
    for i in range(total.size()[0]):
        L2_distance[i, :] = torch.sum((total[i, :] - total) ** 2, 1)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


def guassian_kernel_no_loop(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    # logging.debug('concatenating source and target matrices')
    total = torch.cat([source, target], dim=0)
    # logging.debug('matrix product')
    m = torch.matmul(total, total.t())
    n_row = m.shape[0]
    # logging.debug('the diagonal elements')
    diagonal = torch.diag(m)
    # logging.debug('expansion by diagonal elements')
    te = diagonal.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)))
    tr = diagonal.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)))

    # logging.debug('computing L2 dist')
    L2_distance = -2 * m + te + tr
    # L2_distance = torch.clamp(L2_distance, 0.0, L2_distance)
    # L2_distance=torch.zeros(int(total.size(0)), int(total.size(0))).cuda()
    # L2_distance = Variable(L2_distance)
    # for i in range(total.size()[0]):
    #     L2_distance[i,:]=torch.sum((total[i,:]-total)**2,1)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


def mmd_rbf_accelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel_no_loop(source, target,
                                      kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i + 1) % batch_size
        t1, t2 = s1 + batch_size, s2 + batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)


def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])  # [32,2048]
    # logging.debug('computing gaussian kernel')
    # kernels=torch.zeros(batch_size*2,batch_size*2)
    kernels = guassian_kernel_no_loop(source, target,
                                       kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    # logging.debug('x & x prime')
    XX = kernels[:batch_size, :batch_size]
    # logging.debug('y & y prime')
    YY = kernels[batch_size:, batch_size:]
    # logging.debug('x & y prime')
    XY = kernels[:batch_size, batch_size:]
    # logging.debug('y & x prime')
    YX = kernels[batch_size:, :batch_size]
    # logging.debug('the mean embedding')
    loss = torch.mean(XX + YY - XY - YX)
    return loss
