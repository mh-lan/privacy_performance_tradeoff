#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def DecentralAggre(w, topology_matrix):
    w_locals = copy.deepcopy(w)
    for client in range(topology_matrix.shape[0]):
        for parameter in w_locals[0].keys():
            for neighbor in np.where(topology_matrix[client, :] == 1)[0]:
                w_locals[client][parameter] = w_locals[client][parameter]+w_locals[neighbor][parameter]
                w_locals[client][parameter] = torch.div(w_locals[client][parameter], neighbor.size+1)

    return w_locals
