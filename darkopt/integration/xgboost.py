try:
    import xgboost.core
    _available = True
except ImportError:
    _available = False

import numpy as np

from darkopt import learning_curve


class XGBoostCallback(object):

    def __init__(self, known_best_score, score_key=None,
                 pruning_prob_thresh=0.05,
                 maximize=False, learning_curve_predictor=None,
                 min_iters_before_prune=10, test_interval=10):
        if not _available:
            raise RuntimeError('XGBoost is not installed on your environment.')

        if maximize:
            known_best_score = -known_best_score
        self.known_best_score = known_best_score

        if learning_curve_predictor is None:
            learning_curve_predictor = learning_curve.EnsembleSamplingPredictor()

        self.evals_result_key = score_key
        self.pruning_prob_thresh = pruning_prob_thresh
        self.maximize = maximize
        self.learning_curve_predictor = learning_curve_predictor
        self.min_iters_to_prune = min_iters_before_prune
        self.test_interval = test_interval

        self.history_iterations = []
        self.history_scores = []
        self.prob_win = None
        self.end_iteration = None

    def __call__(self, env):
        if np.isinf(self.known_best_score):
            return

        current_iteration = env.iteration
        if self.evals_result_key is None:
            current_score = env.evaluation_result_list[-1][1]
        else:
            current_score = dict(env.evaluation_result_list)[self.evals_result_key]
        if self.maximize:
            current_score = -current_score

        self.history_iterations.append(current_iteration)
        self.history_scores.append(current_score)
        self.end_iteration = env.end_iteration

        if current_iteration < self.min_iters_to_prune:
            return
        if (current_iteration - self.min_iters_to_prune) % self.test_interval != 0:
            return

        lcp = self.learning_curve_predictor
        lcp.fit(self.history_iterations, self.history_scores)
        self.prob_win = lcp.predict_proba_less_than(env.end_iteration, self.known_best_score)
        print('Probability to beat the known best score:', self.prob_win)
        if self.prob_win < self.pruning_prob_thresh:
            raise xgboost.core.EarlyStopException(env.iteration)

    def info(self):
        if self.prob_win is None:
            return {'pruned': False}

        estimated_score = np.mean(self.learning_curve_predictor.predict_samples(self.end_iteration))
        if self.maximize:
            estimated_score = -estimated_score

        info = {
            'pruned': self.prob_win < self.pruning_prob_thresh,
            'pruning': {
                'estimated_prob_win': self.prob_win,
                'estimated_score': estimated_score,
            },
        }
        if info['pruned']:
            n = self.history_iterations[-1]
            info['pruning']['n_trained_iterations'] = n
            info['pruning']['n_pruned_iterations'] = self.end_iteration - n

        return info
