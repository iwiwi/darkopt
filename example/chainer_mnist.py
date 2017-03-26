#!/usr/bin/env python

import argparse
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import numpy as np

import darkopt

test_batch_size = 1024

param_space = {
    'unit': 2 ** np.arange(3, 10),
    'batch_size': 2 ** np.arange(3, 10),
    'lr': 0.5 ** np.arange(1, 20),
}


class Model(chainer.Chain):

    def __init__(self, n_units):
        super(Model, self).__init__(
            l1=L.Linear(None, n_units),
            l2=L.Linear(None, n_units),
            l3=L.Linear(None, 10),
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--n_trials', '-n', type=int, default=10,
                        help='Number of trials for hyper-parameter search')
    args = parser.parse_args()
    train, test = chainer.datasets.get_mnist()

    def eval_func(param, known_best_score):
        print(param)
        model = L.Classifier(Model(param['unit']))

        if args.gpu >= 0:
            chainer.cuda.get_device(args.gpu).use()
            model.to_gpu()

        optimizer = chainer.optimizers.Adam(param['lr'])
        optimizer.setup(model)

        train_iter = chainer.iterators.SerialIterator(train, param['batch_size'])
        test_iter = chainer.iterators.SerialIterator(
            test, test_batch_size, repeat=False, shuffle=False)
        updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)

        # Trigger for pruning
        pruned_stop_trigger = darkopt.ChainerTrigger(
            # Key for the score to watch
            score_key='validation/main/loss',
            # If there's little chance to beat this score, we prune the training
            known_best_score=known_best_score,
            # Standard training termination trigger
            stop_trigger=(args.epoch, 'epoch')
        )

        trainer = training.Trainer(updater, pruned_stop_trigger, out=args.out)
        trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
        log_report_extension = extensions.LogReport()
        trainer.extend(log_report_extension)
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
        trainer.extend(extensions.ProgressBar())

        trainer.run()
        return log_report_extension.log[-1]['validation/main/loss']

    opt = darkopt.Optimizer(eval_func, param_space)
    best_trial_result = opt.optimize(args.n_trials)
    print('Best validation loss:', best_trial_result.score)
    print('Best parameter:', best_trial_result.param)


if __name__ == '__main__':
    main()
