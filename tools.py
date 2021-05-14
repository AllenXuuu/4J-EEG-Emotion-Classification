import argparse
import time
import os
import numpy as np
import torch


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--svm_C', default=0.01, type=float)
    parser.add_argument('--svm_decision', default='ovo', type=str, choices=['ovo', 'ovr'])
    parser.add_argument('--svm_kernel', default='rbf', type=str, choices=['rbf', 'linear'])
    parser.add_argument('--svm_norm', default=False, action='store_true')

    parser.add_argument('--IDN_weight', default=None, type=str)
    parser.add_argument('--IDN_LSTM_weight', default=None, type=str)

    parser.add_argument('--rep_dim', type=int, default=256)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--early_stop_criterion', type=float, default=1e-3)

    parser.add_argument('--mmd_size', type=int, default=600)
    parser.add_argument('--lambda_rec', type=float, default=0)
    parser.add_argument('--lambda_dom', type=float, default=0)
    parser.add_argument('--lambda_cross', type=float, default=0)
    parser.add_argument('--lambda_mmd', type=float, default=0)
    parser.add_argument('--lambda_cls', type=float, default=0)

    return parser


def evaluate_acc(pred, gt):
    return np.mean(pred == gt)



def store_weight(name, args, report, weights):
    name = name.replace('.py', '')
    folder = 'weights'
    os.makedirs(folder, exist_ok=True)
    # file_name = name + '_' + time.strftime("%m%d_%H%M%S")
    file_name = name
    file_path = os.path.join(folder, file_name)
    print('Writing report to ==> %s.txt' % file_path)
    with open(file_path + '.txt', 'w') as f:
        f.write('Args\n')
        f.write("%s\n\n" % args)
        for line in report:
            f.write(line)
            f.write('\n')
    print('Store weights to ==> %s.pth' % file_path)
    torch.save(weights, file_path + '.pth')
