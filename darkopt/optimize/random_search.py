import numpy as np
import random


def _sample_param(param_space):
    return {
        key: random.choice(space)
        for key, space in param_space.items()
    }


def _random_search(param_space, eval_func, n_calls, prune):
    known_best_score = np.inf
    known_best_param = None

    for _ in range(n_calls):
        param = _sample_param(param_space)
        score = eval_func(param, known_best_score=known_best_score if prune else np.inf)
        print(param, score, known_best_score)
        known_best_score = min(known_best_score, score)

    return known_best_param, known_best_score


def pruned_random_search(param_space, eval_func, n_calls):
    return _random_search(param_space, eval_func, n_calls, True)


def random_search(param_space, eval_func, n_calls):
    return _random_search(param_space, eval_func, n_calls, False)
