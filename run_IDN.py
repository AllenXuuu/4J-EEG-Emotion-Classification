from datautils import Dataloader
import numpy as np
from tools import *
from models import IntegratingDecomposingNetwork
import torch
from collections import defaultdict

if __name__ == '__main__':

    parser = make_parser()
    args = parser.parse_args()
    print(args)
    dataloader = Dataloader()

    pretrained_weights = []
    final_report = []
    print('------Start training------')
    for i in range(15):
        model = IntegratingDecomposingNetwork(args)
        model = model.cuda()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

        for epoch in range(1, args.epoch + 1):

            #################### train
            loss_dict_epoch = defaultdict(list)
            model.train()
            train_feature, train_label = dataloader.get_train_data(i)
            feat_emotion, loss_dict = model(
                torch.from_numpy(train_feature).float().cuda(),
                is_training=True
            )

            total_loss = sum(loss_dict.values())
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            for k, v in loss_dict.items():
                loss_dict_epoch[k].append(v)

            loss_dict_epoch = {
                k: sum(vlist) / len(vlist)
                for k, vlist in loss_dict_epoch.items()
            }

            #################### eval
            if epoch == args.epoch:
                report_str = 'Fold %d epoch %d' % (i, epoch)
                for key in ['loss_rec', 'loss_dom', 'loss_cross', 'loss_mmd']:
                    if key not in loss_dict_epoch:
                        continue
                    loss = loss_dict_epoch[key]
                    report_str += ' %s %.4f' % (key, loss)
                print(report_str)

        pretrained_weights.append(model.state_dict())
        final_report.append(report_str)

    store_weight(__file__, args=args, report=final_report, weights=pretrained_weights)
