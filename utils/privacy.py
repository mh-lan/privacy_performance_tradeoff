import torch
import matplotlib.pyplot as plt
import numpy as np


def imag_distribution(args, dataset_train, dict_users, client_digits, imag_type):
    if imag_type == 'transmitted data' and args.share_ratio == 0:
        client_distribution = [torch.zeros(28, 28) for i in range(args.num_users)]
    else:
        client_distribution = []
        plt.figure()
        for client in range(args.num_users):  # normalized local data distribution and print graph
            img_sum = torch.sum(dataset_train.data[dict_users[client], :, :], dim=0).view([28, 28])
            intensity_distribution = torch.div(img_sum, torch.sum(img_sum))
            img_normalized = intensity_distribution * 255  # .type(torch.int)
            client_distribution.append(intensity_distribution)

            plt.subplot(2, 3, client + 1)
            plt.imshow(img_normalized)
            plt.title('client ' + str(client + 1) + ': ' + str(client_digits[client]))
    if imag_type == 'original':
        plt.savefig('./image distribution/Original  with ' + str(args.digit_num) + ' images')
    elif imag_type == 'exchanged':
        plt.savefig('./image distribution/Exchanged with ' + str(args.digit_num) + ' images ' + str(
            args.share_ratio) + 'share ratio.png')
    elif imag_type == 'transmitted data' and args.share_ratio != 0:
        plt.savefig('./image distribution/Transmitted data with ' + str(args.digit_num) + ' images ' + str(
            args.share_ratio) + ' share ratio.png')
    else:
        print("No transmitted data.")
    return client_distribution


def kl_divergence(distribution_a, distribution_b):
    eps = 1e-5
    distribution_a_normalized = distribution_a + eps * torch.ones(distribution_a.shape)
    distribution_b_normalized = distribution_b + eps * torch.ones(distribution_b.shape)
    return torch.sum(
        distribution_a_normalized * (torch.log(distribution_a_normalized) - torch.log(distribution_b_normalized)))
