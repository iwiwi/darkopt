_status_strs = ['ok', 'error']


class TrialResult(object):

    def __init__(self, score=None, status='ok', info=None):
        if status not in _status_strs:
            raise ValueError(
                "status should be one of the following: {}".format(
                    ', '.join(_status_strs)))

        if status == 'ok':
            score = float(score)

        if info is None:
            info = {}

        self.score = score
        self.status = status
        self.info = info
