
class NoStopping(object):
    def __call__(self, state):
        return False

from .early_stopping import EarlyStopping
