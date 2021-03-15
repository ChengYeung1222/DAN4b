#!/usr/bin/env python
# encoding: utf-8

import torch
from torch.autograd import Variable
import numpy as np

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


def mk_mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])  # [32,2048]
    # logging.debug('computing gaussian kernel')
    # kernels=torch.zeros(batch_size*2,batch_size*2)
    kernels = guassian_kernel_no_loop(source, target,
                                      kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    return kernels


def distance_kernel_beta(coordinate_source, coordinate_target, kernel_mul=2., kernel_num=5, fix_sigma=None):
    n_samples = int(coordinate_source.size()[0]) + int(coordinate_target.size()[0])
    total = torch.cat([coordinate_source, coordinate_target], dim=0)
    m = torch.matmul(total, total.t())
    n_row = m.shape[0]
    # logging.debug('the diagonal elements')
    diagonal = torch.diag(m)
    # logging.debug('expansion by diagonal elements')
    te = diagonal.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)))
    tr = diagonal.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)))

    # logging.debug('computing L2 dist')
    L2_distance = -2 * m + te + tr

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        try:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        except RuntimeError:
            bandwidth = torch.sum(L2_distance) / (n_samples ** 2 - n_samples)
    # bandwidth /= kernel_mul ** (kernel_num // 2)
    # bandwidth /= kernel_mul * (1 / kernel_num)
    bandwidth *= kernel_mul ** kernel_num
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp if bandwidth_temp != 0 else -L2_distance) for bandwidth_temp
                  in bandwidth_list]
    return sum(kernel_val)


def distance_kernel(source, target, coordinate_source, coordinate_target, ):
    n_samples = int(coordinate_source.size()[0]) + int(coordinate_target.size()[0])
    total = torch.cat([coordinate_source, coordinate_target], dim=0)
    k_mmd = mk_mmd(source, target)
    k_dist = torch.zeros(int(total.size(0)), int(total.size(0))).cuda()
    k_dist = Variable(k_dist)

    def Sigma_solution_wo_loop(lr=1e-2, total=total, k_mmd=k_mmd):
        Sigma = Variable(torch.eye(3), requires_grad=True).float().cuda()
        total = total.cuda()

        k_mmd = k_mmd.cuda()
        m = torch.matmul(total, Sigma)
        m = torch.matmul(m, total.t())

        diagonal = torch.diag(m)
        te = diagonal.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)))
        tr = diagonal.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)))

        gaussian = -2 * m + te + tr
        left = -k_mmd * torch.exp(-gaussian)
        nabla_Sigma = Variable(torch.zeros(3, 3)).cuda()
        for u in range(3):
            for v in range(u, 3):
                A = torch.matmul(total[:, u].contiguous().view(int(total.size(0)), 1),
                                 total[:, v].contiguous().view(1, int(total.size(0))))
                B = torch.diag(A).unsqueeze(0).expand(int(total.size(0)), int(total.size(0))) - A - A.t() + torch.diag(
                    A).unsqueeze(1).expand(int(total.size(0)), int(total.size(0)))
                H = left * B
                if u == v:
                    nabla_Sigma[u, v] = torch.mean(H)
                else:
                    nabla_Sigma[u, v] = torch.mean(H)
                    nabla_Sigma[v, u] = torch.mean(H)
            norm_nabla = torch.norm(nabla_Sigma).cpu().data.numpy()
            logging.debug('norm_nabla:%s' % (norm_nabla))
            if norm_nabla < 0.1:  # todo
                break

            Sigma.data -= nabla_Sigma.data * lr
            # print(Sigma)
            eigs_s = np.linalg.eigvals(Sigma.data.cpu())
            logging.debug('eigs_s:%s' % (eigs_s))
            # print(eigs_s)
            # print('{},{}').format(count, Sigma)

        return Sigma

    # total = (total - total.mean()) / total.std()
    def Sigma_solution(lr=1e-2, total=total, k_mmd=k_mmd):
        Sigma = Variable(torch.eye(3), requires_grad=True).float().cuda()
        total = total.cuda()

        k_mmd = k_mmd.cuda()
        count = 0
        while True:
            nabla_k = torch.zeros(int(total.size(0)), int(total.size(0)), 3, 3).cuda()
            nabla_k = Variable(nabla_k)
            for i in range(total.size()[0]):
                for j in range(total.size()[0]):
                    diff = total[i, :] - total[j, :]
                    if diff.data.std() != 0:
                        diff = (diff - diff.mean()) / diff.std()

                        gaussian = torch.matmul(diff.view(1, 3), Sigma)
                        gaussian = torch.matmul(gaussian, diff.view(3, 1))
                        gaussian = -k_mmd[i, j] * torch.exp(-gaussian)
                        # epsilon = Variable(torch.eye(3)).cuda()
                        deri_right = torch.matmul(diff.view(3, 1),
                                                  diff.view(1, 3))  # + 0.001 * epsilon
                        gaussian = gaussian * deri_right
                        # eigs = np.linalg.eigvals(deri_right.data.cpu())
                        # if np.all(eigs > 0):
                        #     print('pd')

                    else:
                        gaussian = torch.eye(3)
                    nabla_k[i, j] = gaussian
            nabla_Sigma = torch.mean(nabla_k.mean(dim=0), dim=0)
            # print(nabla_Sigma)
            norm_nabla = torch.norm(nabla_Sigma).cpu().data.numpy()
            logging.debug('norm_nabla:%s' % (norm_nabla))
            if norm_nabla < 0.1:  # todo
                break

            Sigma.data -= nabla_Sigma.data * lr
            count += 1
            logging.debug('%dth gd...' % (count))
            # print(Sigma)
            eigs_s = np.linalg.eigvals(Sigma.data.cpu())
            logging.debug('eigs_s:%s' % (eigs_s))
            # print(eigs_s)
            # print('{},{}').format(count, Sigma)
        return Sigma

    Sigma = Sigma_solution_wo_loop()
    # Sigma = Variable(torch.eye(3), requires_grad=True).float().cuda()  # todo:debug dist kernel

    for i in range(total.size()[0]):
        for j in range(total.size()[0]):
            diff = total[i, :] - total[j, :]
            if diff.data.std() != 0:
                diff = (diff - diff.mean()) / diff.std()
                gaussian = torch.matmul(diff.view(1, 3), Sigma)
                gaussian = torch.matmul(gaussian, diff.view(3, 1))
                gaussian = torch.exp(-gaussian)
            else:
                gaussian = Variable(torch.exp(torch.zeros(1))).cuda()
            k_dist[i, j] = gaussian
    return k_dist


def guassian_kernel_no_loop(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    # logging.debug('concatenating source and target matrices')
    # if len(source.shape) != 2:
    #     source=source.data.resize(source.shape[0], source.shape[1] * source.shape[2] * source.shape[3])
    #     target=target.data.resize(target.shape[0], target.shape[1] * target.shape[2] * target.shape[3])
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
        try:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        except RuntimeError:
            bandwidth = torch.sum(L2_distance) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp if bandwidth_temp != 0 else -L2_distance) for bandwidth_temp
                  in bandwidth_list]
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


def mmd_rbf_noaccelerate(source, target, coordinate_source, coordinate_target, kernel_i, heterogeneity,
                         # fd_kernel=False,
                         kernel_mul=2.0, kernel_num=5,
                         fix_sigma=None, Lambda=1):
    batch_size = int(source.size()[0])  # [32,2048]
    # logging.debug('computing gaussian kernel')
    # kernels=torch.zeros(batch_size*2,batch_size*2)
    kernels = guassian_kernel_no_loop(source, target,
                                      kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    if heterogeneity == True:
        # kernels_dist = distance_kernel(source, target, coordinate_source, coordinate_target)
        kernels_dist = distance_kernel_beta(coordinate_source, coordinate_target)
        for i in range(batch_size * 2):
            for j in range(batch_size * 2):
                if torch.abs(kernels_dist[i, j]).cpu().data.numpy() > 2e+1:
                    kernels_dist[i, j] = 0

        XX = kernels[:batch_size, :batch_size] + kernels[:batch_size, :batch_size] * \
             kernels_dist[:batch_size, :batch_size] * Lambda
        # todo:核都是1
        # Variable(torch.ones(batch_size,
        #                     batch_size)).cuda()  # \
        YY = kernels[batch_size:, batch_size:] + kernels[batch_size:, batch_size:] * \
             kernels_dist[batch_size:, batch_size:] * Lambda
        # Variable(torch.ones(batch_size,
        #                     batch_size)).cuda()  # \
        XY = kernels[:batch_size, batch_size:] + kernels[:batch_size, batch_size:] * \
             kernels_dist[:batch_size, batch_size:] * Lambda

        YX = kernels[batch_size:, :batch_size] + kernels[batch_size:, :batch_size] * \
             kernels_dist[batch_size:, :batch_size] * Lambda
    # elif fd_kernel == True:
    #     # logging.debug('x & x prime')
    #     XX = kernels[:batch_size, :batch_size] * Variable(
    #         kernel_i[:batch_size, :batch_size])
    #     # logging.debug('y & y prime')
    #     YY = kernels[batch_size:, batch_size:] * Variable(
    #         kernel_i[batch_size:, batch_size:])
    #     # logging.debug('x & y prime')
    #     XY = kernels[:batch_size, batch_size:] * Variable(
    #         kernel_i[:batch_size, batch_size:])
    #     # logging.debug('y & x prime')
    #     YX = kernels[batch_size:, :batch_size] * Variable(
    #         kernel_i[batch_size:, :batch_size])
    #     # logging.debug('the mean embedding')
    else:
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    # logging.debug('kernel loss with dist = %s' % (loss))
    return loss
