try:
    import chainer.training
    _available = True
    _base = chainer.training.IntervalTrigger
except ImportError:
    _available = False
    _base = object

import numpy as np

from darkopt import learning_curve


class ChainerPruningTrigger(_base):

    # TODO(iwiwi): explain that this class inherits IntervalTrigger because of ProgressBar

    def __init__(self, score_key, known_best_score, stop_trigger,
                 maximize=False, test_trigger=(5, 'epoch'),
                 pruning_prob_thresh=0.05, learning_curve_predictor=None):
        if not _available:
            raise RuntimeError('Chainer is not installed on your environment.')

        stop_trigger = chainer.training.get_trigger(stop_trigger)
        test_trigger = chainer.training.get_trigger(test_trigger)
        assert isinstance(stop_trigger, chainer.training.IntervalTrigger)
        assert isinstance(test_trigger, chainer.training.IntervalTrigger)
        assert stop_trigger.unit == test_trigger.unit
        super(ChainerPruningTrigger, self).__init__(stop_trigger.period, stop_trigger.unit)

        if learning_curve_predictor is None:
            learning_curve_predictor = learning_curve.EnsembleSamplingPredictor()

        self.stop_trigger = stop_trigger
        self.test_trigger = test_trigger
        self.score_key = score_key
        self.known_best_score = known_best_score
        self.maximize = maximize
        self.pruning_prob_thresh = pruning_prob_thresh
        self.learning_curve_predictor = learning_curve_predictor

        self.history_iterations = []
        self.history_scores = []

    def __call__(self, trainer):
        if self.stop_trigger(trainer):
            return True

        if np.isinf(self.known_best_score):
            return False

        observation = trainer.observation
        if self.score_key in observation:
            current_iteration = getattr(trainer.updater, self.test_trigger.unit)
            current_score = float(observation[self.score_key])
            self.history_iterations.append(current_iteration)
            self.history_scores.append(current_score)

        if not self.test_trigger(trainer):
            return False

        lcp = self.learning_curve_predictor
        lcp.fit(self.history_iterations, self.history_scores)
        prob_win = lcp.predict_proba_less_than(self.stop_trigger.period, self.known_best_score)
        print('Probability to beat the known best score:', prob_win)
        return prob_win < self.pruning_prob_thresh
