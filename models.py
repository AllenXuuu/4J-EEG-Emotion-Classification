import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math

LENGTH = 3394


########################################### IDN
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(LENGTH),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(LENGTH),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layer(x) + x


class Integrator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Integrator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.layers = nn.Sequential(
            nn.Linear(2 * in_dim, in_dim),
            ResidualBlock(in_dim),
            ResidualBlock(in_dim),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, feat_domain, feat_emotion):
        composition = torch.cat([feat_domain, feat_emotion], -1)
        return self.layers(composition)


class Decomposer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Decomposer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.layers = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            ResidualBlock(out_dim),
            ResidualBlock(out_dim),
            nn.Linear(out_dim, 2 * out_dim)
        )

    def forward(self, composition):
        composition = self.layers(composition)
        feat_domain, feat_emotion = torch.split(composition, self.out_dim, -1)
        return feat_domain, feat_emotion


class IntegratingDecomposingNetwork(nn.Module):
    def __init__(self, args):
        super(IntegratingDecomposingNetwork, self).__init__()
        self.length = 3394
        self.decomposer = Decomposer(310, args.rep_dim)
        self.integrator = Integrator(args.rep_dim, 310)
        self.args = args

    def mmd(self, source, target, kernel_mul=2.0, kernel_num=5, sigma=1):
        length, dim = source.shape

        total = torch.cat([source, target], dim=0)

        total_exp0 = total.unsqueeze(0).expand(2 * length, 2 * length, dim)
        total_exp1 = total.unsqueeze(1).expand(2 * length, 2 * length, dim)

        dist = (total_exp0 - total_exp1).pow_(2).sum(-1)
        bandwidth = sigma / (kernel_mul ** (kernel_num // 2))
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernels = [torch.exp(-dist / bandwidth) for bandwidth in bandwidth_list]
        kernels = sum(kernels)

        XX = kernels[:length, :length]
        YY = kernels[length:, length:]
        XY = kernels[:length, length:]
        YX = kernels[length:, :length]
        loss = torch.mean(XX + YY - XY - YX)

        return loss

    def forward(self, EEG_feature, is_training=False):
        domains, length, _ = EEG_feature.shape

        device = EEG_feature.device
        feat_domain, feat_emotion = self.decomposer(EEG_feature)

        ##################### test
        if not is_training:
            return feat_emotion
        ##################### train
        total_loss = {}
        ###### reconstruction MSE
        if self.args.lambda_rec > 0:
            feat_rec = self.integrator(feat_domain, feat_emotion)
            loss_rec = self.args.lambda_rec * F.mse_loss(feat_rec, EEG_feature)
            total_loss.update({
                'loss_rec': loss_rec
            })
        ###### dom feature MSE
        feat_domain_randomInDomain = torch.index_select(
            feat_domain,
            dim=1,
            index=torch.from_numpy(np.random.permutation(length)).to(device)
        )
        if self.args.lambda_dom > 0:
            loss_dom = self.args.lambda_dom * F.mse_loss(feat_domain_randomInDomain, feat_domain)
            total_loss.update({
                'loss_dom': loss_dom
            })
        ###### cross MSE
        if self.args.lambda_cross > 0:
            feat_rec = self.integrator(feat_domain_randomInDomain, feat_emotion)
            loss_cross = self.args.lambda_cross * F.mse_loss(feat_rec, EEG_feature)
            total_loss.update({
                'loss_cross': loss_cross
            })
        ###### MMD loss
        if self.args.lambda_mmd > 0:
            index = torch.randperm(LENGTH)[:self.args.mmd_size].to(feat_emotion.device)
            loss_mmd = self.args.lambda_mmd * self.mmd(feat_emotion[0, index], feat_emotion[1, index])
            total_loss.update({
                'loss_mmd': loss_mmd
            })
        return feat_emotion, total_loss


class IDN_LSTM(nn.Module):
    def __init__(self, args):
        super(IDN_LSTM, self).__init__()
        self.length = 3394
        self.args = args

        self.IDN = IntegratingDecomposingNetwork(args)
        self.postLSTM = nn.LSTM(
            input_size=args.rep_dim,
            hidden_size=args.rep_dim,
            num_layers=3,
            batch_first=True,
        )
        self.classifier = nn.Linear(args.rep_dim, 3)

    def forward(self, EEG_feature, label=None, is_training=False):
        domains, length, _ = EEG_feature.shape
        device = EEG_feature.device
        feat_emo = self.IDN(EEG_feature, is_training=is_training)
        if type(feat_emo) == tuple:
            feat_emo = feat_emo[0]
        feat_lstm, _ = self.postLSTM(feat_emo)
        score = self.classifier(feat_lstm + feat_emo)
        ##################### test
        if not is_training:
            return feat_emo, score
        ##################### train
        total_loss = {}

        if self.args.lambda_cls > 0:
            loss_cls = nn.functional.cross_entropy(score.transpose(1, 2), label.long())
            loss_cls = loss_cls * self.args.lambda_cls
            total_loss.update({
                'loss_cls': loss_cls
            })
        return feat_emo, score, total_loss

    def load_IDN_weight(self, weight):
        self.IDN.load_state_dict(weight)
