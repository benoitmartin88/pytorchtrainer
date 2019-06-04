from trainer import ModuleTrainer


class Callback(object):
    def __call__(self, trainer: ModuleTrainer):
        raise NotImplementedError()
