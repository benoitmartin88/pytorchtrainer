from .early_stopping import EarlyStopping


class NoStopping(object):
    def __call__(self, state):
        return False
