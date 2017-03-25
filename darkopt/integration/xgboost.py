import numpy as np
import xgboost.core

from darkopt import learning_curve


class XGBoostCallback(object):

    def __init__(self, known_best_score, score_key=None,
                 pruning_prob_thresh=0.05,
                 maximize=False, learning_curve_predictor=None,
                 min_iters_before_prune=10, test_interval=10):
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

        if current_iteration < self.min_iters_to_prune:
            return
        if (current_iteration - self.min_iters_to_prune) % self.test_interval != 0:
            return

        lcp = self.learning_curve_predictor
        lcp.fit(self.history_iterations, self.history_scores)
        prob_win = lcp.predict_proba_less_than(env.end_iteration, self.known_best_score)
        print('Probability to beat the known best score:', prob_win)
        if prob_win < self.pruning_prob_thresh:
            raise xgboost.core.EarlyStopException(env.iteration)
