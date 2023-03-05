import numpy as np


# class for holding the random state throughout the notebook.
# this keeps results consistent
class RandomState(object):
    def __init__(self, random_state=None):
        self.random_state = random_state

    def next(self, n=1):
        assert type(n) == int, "Ensure n is an integer"
        if n == 1:
            self.random_state, out_state = np.random.default_rng(
                self.random_state
            ).integers(0, 1e9, size=(2,))
        else:
            self.random_state, *out_state = np.random.default_rng(
                self.random_state
            ).integers(0, 1e9, size=(n + 1,))

        return out_state
