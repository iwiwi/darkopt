import random


def _sample_param(param_space):
    return {
        key: random.choice(space)
        for key, space in param_space.items()
    }


class RandomSearch(object):

    def __init__(self, param_space):
        self.param_space = param_space

    def suggest(self):
        return _sample_param(self.param_space)

    def report(self, param, result):
        pass
