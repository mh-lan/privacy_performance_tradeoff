#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = np.random.choice(all_idxs, num_items, replace=False)
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    client_digits = ['all' for i in range(num_users)]
    return dict_users, client_digits


def mnist_noniid(dataset, num_users, digit_num):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 10, 6000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.targets.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))  # indicate label of each img
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  # rearrange according to label
    idxs = idxs_labels[0, :]

    # divide and assign
    client_digits = [None for i in range(num_users)]
    included_digits = set()
    for i in range(num_users):
        client_digits[i] = set(
            [(i * num_users + digit) % 10 for digit in range(i, i + digit_num)])  # randomly choosing shards
        included_digits = included_digits | client_digits[i]
        for rand in client_digits[i]:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
        print('training digits of client %d is %s' % (i, str(client_digits[i])))
    return dict_users, client_digits


def data_exchange(args, dict_users_original, topology_matrix):
    dict_users = {i: dict_users_original[i] for i in range(args.num_users)}
    num_imgs = dict_users_original[0].size
    transmit_data = [None for i in range(args.num_users)]
    for i in range(args.num_users):
        data_for_exchange = np.random.choice(num_imgs, int(num_imgs * args.share_ratio), replace=False)
        transmit_data[i] = dict_users_original[i][data_for_exchange]
        for j in np.where(topology_matrix[i, :] == 1)[0]:
            dict_users[j] = np.concatenate((dict_users[j], transmit_data[i]), axis=0)

    for i in range(args.num_users):
        dict_users[i] = np.unique(dict_users[i], axis=0)
        np.random.shuffle(dict_users[i])  # delete repeated items and shuffle

    return dict_users, transmit_data


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 5
    digit_num = 4
    share_ratio = 0.2
    d = mnist_noniid(dataset_train, num, digit_num)
    topology_matrix = np.array([[0., 1., 1., 1., 0.],
                            [1., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 1.],
                            [1., 0., 0., 0., 0.],
                            [0., 0., 1., 0., 0.]])
    d = data_exchange(d, topology_matrix, share_ratio, num)
