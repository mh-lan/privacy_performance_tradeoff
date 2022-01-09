#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import sys
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import copy
import torch
import numpy as np
from torchvision import datasets, transforms

from utils.sampling import mnist_iid, mnist_noniid, data_exchange
from utils.options import args_parser
from utils.privacy import imag_distribution, kl_divergence

from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist
from models.Fed import FedAvg, DecentralAggre
from models.test import test_img

# from torchviz import make_dot

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # create folder for result
    dirs = ['./image distribution', './save']
    for dir_result in dirs:
        if not os.path.exists(dir_result):
            os.makedirs(dir_result)

    # create log file
    log = open('result.log', "a")
    # sys.stdout = log

    matplotlib.use('Agg')
    print('The %d training setting starts!' % args.excel_row)
    print("Model: %s, Epoch: %d, iid: %d, aggregating method: %s, share_ratio: %f, original digit num: %d" % (
        args.model, args.epochs, args.iid, args.aggr_model, args.share_ratio,
        args.digit_num))

    # create Excel file to store result
    excel_title = ['nn model', 'share ratio', 'number of original digits', 'aggregation model', 'iid status',
                   'training rounds',
                   'acc client 1', 'acc client 2', 'acc client 3', 'acc client 4', 'acc client 5',
                   'leakage 1', 'leakage 2', 'leakage 3', 'leakage 4', 'leakage 5']
    if not os.path.exists(args.excel_file):
        data = pd.DataFrame(columns=excel_title)
    else:
        data = pd.read_excel(args.excel_file)
        data.columns = excel_title  # , index=excel_index)

    # write simulation parameters in Excel file
    parameters = [args.model, args.share_ratio, args.digit_num, args.aggr_model, args.iid, args.epochs, args.gpu]
    for idx in range(len(parameters) - 1):
        data.loc[args.excel_row, excel_title[idx]] = parameters[idx]
    data.to_excel(args.excel_file, index=False)

    # load MNIST dataset and split users
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)

    # create training data for users according to iid/non-iid setting
    if args.iid:
        dict_users_original, client_digits = mnist_iid(dataset_train, args.num_users)
    else:
        dict_users_original, client_digits = mnist_noniid(dataset_train, args.num_users, args.digit_num)
    img_size = dataset_train[0][0].shape

    # build topology
    link_status = [[0, 1], [0, 2], [0, 3], [2, 4]]
    topology_matrix = np.zeros([args.num_users, args.num_users])
    for ind in link_status:
        topology_matrix[ind[0], ind[1]] = 1
    topology_matrix = topology_matrix + topology_matrix.T  # record neighbors only

    # data exchange
    dict_users, transmit_data = data_exchange(args, dict_users_original, topology_matrix)

    # average intensity of each pixel
    client_original_distribution = imag_distribution(args, dataset_train, dict_users_original, client_digits,
                                                     'original')
    client_exchanged_distribution = imag_distribution(args, dataset_train, dict_users, client_digits, 'exchanged')
    client_transmit_distribution = imag_distribution(args, dataset_train, transmit_data, client_digits,
                                                     'transmitted data')

    # privacy leakage is defined as KL divergence of transmit data and receiver data distribution
    privacy_leakage = [None for i in range(args.num_users)]
    for client in range(args.num_users):
        divergence_neighbors = 0
        for neighbor in np.where(topology_matrix[client, :] != 0)[0]:
            divergence_neighbors = divergence_neighbors + kl_divergence(client_transmit_distribution[client],
                                                                        client_exchanged_distribution[neighbor])
        privacy_leakage[client] = divergence_neighbors

    privacy_leakage = [args.share_ratio * privacy_leakage[i] for i in range(args.num_users)]
    privacy_leakage_sum = sum(privacy_leakage)

    # write privacy leakage in Excel file
    for idx in range(len(privacy_leakage)):
        print("Privacy leakage of client %d: %f" % (idx, privacy_leakage[idx]))
        data.loc[args.excel_row, 'leakage ' + str(idx + 1)] = privacy_leakage[idx].numpy()
    data.to_excel(args.excel_file, index=False)

    # build nn model
    if args.model == 'cnn':
        net_train = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_train = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')

    # print model via torchviz
    # install torchviz and graphviz first
    print(net_train)  # only one net to be trained
    if args.plot_model is True:
        y = torch.rand([1, 1, 28, 28])
        g = make_dot(net_train(y))  # print model structure via torchviz
        g.render(args.model + '_model', view=False)

    # train nn model
    net_train.train()

    # initiate each client's weights
    w_initial = net_train.state_dict()
    w_locals = [w_initial for i in range(args.num_users)]

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    for iter_epoch in range(args.epochs):
        loss_locals = []

        for idx in range(args.num_users):
            # training each client's parameter via net_train
            net_train.load_state_dict(w_locals[idx])  # load parameters
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_train).to(args.device))

            w_locals[idx] = copy.deepcopy(w)  # update trained local parameter
            loss_locals.append(copy.deepcopy(loss))

        # updating clients' parameters via 2 aggregation methods
        if args.aggr_model == 'ps':
            # aggregate local model as global weights
            w_glob = FedAvg(w_locals)
            # broadcasting global weights
            w_locals = [w_glob for i in range(args.num_users)]
        elif args.aggr_model == 'de':
            # update parameter of each client
            w_locals = DecentralAggre(w_locals, topology_matrix)
        else:
            print("Error: undefined aggregation model")

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter_epoch, loss_avg))
        loss_train.append(loss_avg)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig(
        './save/fed_{}_{}_iid{}_{}_{}_digit{}.png'.format(args.model, args.epochs, args.iid, args.aggr_model,
                                                          args.share_ratio,
                                                          args.digit_num))

    # testing
    net_train.eval()

    if args.aggr_model == 'ps':
        acc_test, loss_test = test_img(net_train, dataset_test, args)
        print("Testing accuracy: {:.2f}".format(acc_test))
        data.loc[args.excel_row, 'acc client 1'] = acc_test.numpy()
        data.to_excel(args.excel_file, index=False)
    else:
        acc_test = [None for i in range(args.num_users)]
        for idx in range(args.num_users):
            net_train.load_state_dict(w_locals[idx])
            acc_test[idx], loss_test = test_img(net_train, dataset_test, args)
            print("Testing accuracy of client %d: %f" % (idx, acc_test[idx]))
            data.loc[args.excel_row, 'acc client ' + str(idx + 1)] = acc_test[idx].numpy()
        data.to_excel(args.excel_file, index=False)
