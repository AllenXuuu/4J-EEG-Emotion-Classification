from datautils import Dataloader
import numpy as np
from tools import *
from models import IDN_LSTM
import torch
from collections import defaultdict

if __name__ == '__main__':

    parser = make_parser()
    args = parser.parse_args()
    print(args)
    dataloader = Dataloader()

    print('Restore ckpt from <==' + args.IDN_weight)
    pretrained_weights = torch.load(args.IDN_weight)
    final_weights = []
    final_report = []
    final_acc = []
    print('------Start training------')
    for i in range(15):
        model = IDN_LSTM(args)
        model.load_IDN_weight(pretrained_weights[i])
        model = model.cuda()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

        for epoch in range(1, args.epoch + 1):
            #################### train
            loss_dict_epoch = defaultdict(list)
            model.train()
            train_feature, train_label = dataloader.get_train_data(i)
            feat_emotion, fc_prob, loss_dict = model(
                torch.from_numpy(train_feature).float().cuda(),
                torch.from_numpy(train_label).float().cuda(),
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
            if loss_dict_epoch['loss_cls'] <= args.early_stop_criterion or epoch == args.epoch:
                #################### eval
                model.eval()
                val_feature, val_label = dataloader.get_val_data(i)
                feat_emotion, fc_prob = model(
                    torch.from_numpy(val_feature).float().cuda(),
                    torch.from_numpy(val_label).float().cuda(),
                    is_training=False
                )
                pred_label = torch.argmax(fc_prob, -1).data.cpu().numpy()

                acc = np.mean(val_label == pred_label)
                final_acc.append(acc)
                report_str = 'Fold %d Epoch %d Acc %.4f. Stop training' % (i, epoch, acc)
                print(report_str)
                break

        final_report.append(report_str)
        final_weights.append(model.state_dict())

    avg_acc = np.mean(final_acc)
    std_acc = np.std(final_acc)
    avgstd_report = 'Acc avg %.4f std %.4f' % (avg_acc, std_acc)
    final_report.append(avgstd_report)
    print(avgstd_report)

    store_weight(__file__, args=args, report=final_report, weights=final_weights)
