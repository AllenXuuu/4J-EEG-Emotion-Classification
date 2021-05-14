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

    print('Restore ckpt from <==' + args.IDN_LSTM_weight)
    pretrained_weights = torch.load(args.IDN_LSTM_weight)

    model = IDN_LSTM(args)
    model = model.cuda()

    final_acc = []
    for i in range(15):
        model.load_state_dict(pretrained_weights[i])
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
        report_str = 'Fold %d Acc %.4f' % (i, acc)
        print(report_str)

    avg_acc = np.mean(final_acc)
    std_acc = np.std(final_acc)
    avgstd_report = 'Acc avg %.4f std %.4f' % (avg_acc, std_acc)
    print(avgstd_report)
