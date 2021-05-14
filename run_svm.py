from datautils import Dataloader
from sklearn.svm import SVC
import numpy as np
from tools import *

if __name__ == '__main__':

    parser = make_parser()
    args = parser.parse_args()
    print(args)
    dataloader = Dataloader()
    acc_all_fold = []
    
    for i in range(15):
        train_feature, train_label = dataloader.get_train_data(i)
        train_feature = train_feature.reshape(-1, train_feature.shape[-1])
        train_label = train_label.flatten()
        val_feature, val_label = dataloader.get_val_data(i)
        val_feature = val_feature.reshape(-1, val_feature.shape[-1])
        val_label = val_label.flatten()

        model = SVC(C=args.svm_C,
                    kernel=args.svm_kernel,
                    decision_function_shape=args.svm_decision,
                    max_iter=100000
                    )
        model.fit(train_feature, train_label)
        val_pred = model.predict(val_feature)

        acc = evaluate_acc(val_pred, val_label)
        acc_all_fold.append(acc)
        print('# Fold-%d, Accuracy: %.4f' % (i, acc))
    print("# Avg: %.4f" % np.mean(acc_all_fold))
    print("# Std: %.4f" % np.std(acc_all_fold))
    print('Finish.')
