import numpy as np

from darkopt.optimize import engine as engine_module
from darkopt.optimize import trial_result as trial_result_module


def _is_better(new_score, known_best_score, maximize):
    if maximize:
        return new_score > known_best_score
    else:
        return new_score < known_best_score


def _get_inf(maximize):
    if maximize:
        return -np.inf
    else:
        return np.inf


class Optimizer(object):

    def __init__(self, target_func, param_space, engine='random_search', maximize=False, prune=True):
        self.target_func = target_func
        self.param_space = param_space
        self.engine = engine_module.get_engine(engine, param_space)
        self.maximize = maximize
        self.prune = prune

        self.trial_results = []
        self.known_best_trial = trial_result_module.TrialResult(_get_inf(maximize))

    def optimize(self, max_n_calls=None):
        while max_n_calls is None or max_n_calls > 0:
            param = self.engine.suggest()

            if self.prune:
                known_best_score = self.known_best_trial.score
            else:
                known_best_score = _get_inf(self.maximize)

            trial_result = self.target_func(param, known_best_score)
            trial_result = trial_result_module.get_trial_result(trial_result)
            trial_result.param = param

            # TODO(iwiwi): engines need to know whether maximizing or minimizing
            self.engine.report(param, trial_result)
            self.trial_results.append(trial_result)
            print(param, trial_result.score, known_best_score)

            if trial_result.status == 'ok' and _is_better(
                    trial_result.score, self.known_best_trial.score, self.maximize):
                self.known_best_trial = trial_result

            if max_n_calls is not None:
                max_n_calls -= 1

        return self.known_best_trial
