import numpy as np

from darkopt.learning_curve import sampling


class SingleSamplingPredictor(object):

    def __init__(self, curve='vapore_pressure'):
        self.curve = curve
        self.traces_ = None

    def fit(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        self.traces_ = sampling.sample_single(x, y, self.curve, None, None)
        return self

    def predict_samples(self, x):
        return [
            sampling.predict_single(x, self.curve, trace)
            for trace in self.traces_
        ]

    def predict_proba_less_than(self, x, y):
        return np.mean([
            sampling.predict_proba_less_than_single(x, y, self.curve, trace)
            for trace in self.traces_
        ], axis=0)

    def predict_proba_greater_than(self, x, y):
        return 1 - self.predict_proba_less_than(x, y)


class EnsembleSamplingPredictor(object):

    def __init__(self, curves='all', map_options=None, sample_options=None):
        self.sample_options = sample_options
        self.map_options = map_options
        self.curves = curves
        self.traces_ = None

    def fit(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        self.traces_ = sampling.sample_ensemble(
            x, y, self.curves, self.sample_options, self.map_options)
        return self

    def predict_samples(self, x):
        return [
            sampling.predict_ensemble(x, self.curves, trace)
            for trace in self.traces_
        ]

    def predict_proba_less_than(self, x, y):
        return np.mean([
            sampling.predict_proba_less_than_ensemble(x, y, self.curves, trace)
            for trace in self.traces_
        ], axis=0)

    def predict_proba_greater_than(self, x, y):
        return 1 - self.predict_proba_less_than(x, y)
