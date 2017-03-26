#!/usr/bin/python

# Based on the XGBoost official example
# (https://github.com/dmlc/xgboost/tree/master/demo/kaggle-higgs)

import argparse
import numpy as np
import sklearn.model_selection
import xgboost

import darkopt

test_size = 550000
val_portion = 0.1

param_space = {
    'max_depth': np.arange(1, 16),
    'eta': 0.5 ** np.arange(1, 20),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='./training.csv',
                        help='Input data')
    parser.add_argument('--n_trials', '-n', type=int, default=10,
                        help='Number of trials for hyper-parameter search')
    args = parser.parse_args()

    dtrain = np.loadtxt(args.input, delimiter=',', skiprows=1,
                        converters={32: lambda x: int(x == 's'.encode('utf-8'))})
    x = dtrain[:, 1:31]
    w = dtrain[:, 31] * float(test_size) / len(x)  # Weight
    y = dtrain[:, 32]
    x_tra, x_val, w_tra, w_val, y_tra, y_val = sklearn.model_selection.train_test_split(
        x, w, y, test_size=val_portion, random_state=0)

    sum_wpos = sum(w_tra[i] for i in range(len(y_tra)) if y_tra[i] == 1.0)
    sum_wneg = sum(w_tra[i] for i in range(len(y_tra)) if y_tra[i] == 0.0)
    dm_tra = xgboost.DMatrix(x_tra, label=y_tra, missing=-999.0, weight=w_tra)
    dm_val = xgboost.DMatrix(x_val, label=y_val, missing=-999.0, weight=w_val)

    def eval_func(param, known_best_score):
        print(param)
        config = {}
        config['objective'] = 'binary:logitraw'
        config['scale_pos_weight'] = sum_wneg / sum_wpos
        config['eta'] = param['eta']
        config['max_depth'] = param['max_depth']
        config['eval_metric'] = 'auc'
        config['silent'] = 1
        config['nthread'] = 16
        num_round = 30

        plst = list(config.items()) + [('eval_metric', 'ams@0.15')]
        watchlist = [(dm_tra, 'train'), (dm_val, 'val')]

        # Callback for pruning
        darkopt_callback = darkopt.XGBoostCallback(
            # Key for the score to watch
            score_key='val-ams@0.15',
            # If there's little chance to beat this score, we prune the training
            known_best_score=known_best_score,
            # We are maximizing the score
            maximize=True,
        )

        evals_result = {}
        xgboost.train(plst, dm_tra, num_round, watchlist,
                      evals_result=evals_result,
                      callbacks=[darkopt_callback])
        return darkopt.optimize.TrialResult(evals_result['val']['ams@0.15'][-1])

    opt = darkopt.optimize.Optimizer(eval_func, param_space, maximize=True)
    best_trial_result = opt.optimize(args.n_trials)
    print('Best validation loss:', best_trial_result.score)
    print('Best parameter:', best_trial_result.param)


if __name__ == '__main__':
    main()
