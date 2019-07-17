from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import struct
import torch
from torchvision import transforms

import logging
import time


def default_loader(img):
    return Image.open(img)


def img_transform():
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),  # todo
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    return transform


class custom_dset(Dataset):
    def __init__(self,
                 txt_path,
                 img_transform=None,
                 n_channels=6,
                 nx=227,
                 nz=227,
                 labeled=True
                 ):
        self.nx = nx
        self.nz = nz
        self.n_channels = n_channels
        self.img_list = []
        self.label_list = []
        self.labeled=labeled
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            # print(len(lines))
            for line in lines:
                # print(line)
                if labeled==True:
                    items = line.split(',')  # .csv
                    self.img_list.append(items[0])
                    self.label_list.append(int(items[1]))
                else:
                    self.img_list.append(line[:-1])
        self.img_transform = img_transform

    def __getitem__(self, index):
        img_path = self.img_list[index]
        if self.labeled:
            label = self.label_list[index]
        # img = self.loader(img_path)
        img = img_path
        img = self._xshow(img)
        if self.img_transform is not None:
            self.img_transform()
        if self.labeled:
            return img, label
        else:
            return img

    def __len__(self):
        return len(self.img_list)

    def _xshow(self, filename):
        nx = self.nx
        nz = self.nz
        n_channels = self.n_channels
        f = open(filename, "rb")
        # logging.info('open file:%s'%(filename))
        pic = np.zeros((nx, nz, n_channels))

        for i in range(nx):
            for j in range(nz):
                for k in range(n_channels):
                    data = f.read(4)
                    elem = struct.unpack("f", data)[0]
                    pic[i][j][k] = elem
        pic = np.swapaxes(pic, 0, 2)

        f.close()
        return pic


def collate_fn(batch):
    batch.sort()
    img, label = zip(*batch)
    return img, label

# dset = custom_dset('./top100_list.csv',labeled=False)
# custom_loader = DataLoader(dset, batch_size=1, shuffle=False,)# collate_fn=collate_fn)
# img = custom_loader.__iter__().__next__()
# # custom_loader.__getitem__().img_path
# iter_custom=iter(custom_loader)
# for i in range(len(custom_loader)):
#
#     data=next(iter_custom)
#
# print(img)
# print(dset[0][0])
