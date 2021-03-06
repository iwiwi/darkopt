_status_strs = ['ok', 'pruned', 'error']


class TrialResult(object):

    def __init__(self, score=None, status='ok', **info):
        if status not in _status_strs:
            raise ValueError(
                "status should be one of the following: {}".format(
                    ', '.join(_status_strs)))

        if status == 'ok':
            score = float(score)

        self.score = score
        self.status = status
        self.info = info
        self.param = None


def get_trial_result(trial_result):
    if isinstance(trial_result, TrialResult):
        return trial_result
    else:
        return TrialResult(trial_result)
