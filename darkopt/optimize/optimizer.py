import numpy as np

from darkopt.optimize import engine as engine_module
from darkopt.optimize import trial_result as trial_result_module


class Optimizer(object):

    def __init__(self, target_func, param_space, engine='random_search', prune=True):
        self.target_func = target_func
        self.param_space = param_space
        self.engine = engine_module.get_engine(engine, param_space)
        self.prune = prune

        self.trial_results = []
        self.known_best_trial = trial_result_module.TrialResult(np.inf)

    def optimize(self, max_n_calls=None):
        while max_n_calls is None or max_n_calls > 0:
            param = self.engine.suggest()

            if self.prune:
                known_best_score = self.known_best_trial.score
            else:
                known_best_score = np.inf

            trial_result = self.target_func(param, known_best_score)
            trial_result = trial_result_module.get_trial_result(trial_result)
            trial_result.param = param
            self.trial_results.append(trial_result)
            print(param, trial_result.score, known_best_score)

            if trial_result.status == 'ok' and trial_result.score < self.known_best_trial.score:
                self.known_best_trial = trial_result

            if max_n_calls is not None:
                max_n_calls -= 1

        return self.known_best_trial
