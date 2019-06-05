
class EarlyStopping(object):
    def __init__(self, patience=10,
                 metric=lambda state: state.last_train_loss,
                 comparison_function=lambda metric, best: metric >= best):
        if patience < 1:
            raise ValueError("Argument patience should be positive integer.")

        if not callable(metric):
            raise TypeError("Argument metric should be a function.")

        if not callable(comparison_function):
            raise TypeError("Argument score_function should be a function.")

        self.patience = patience
        self.metric = metric
        self.comparison_function = comparison_function

        self.counter = 0
        self.best_score = None

    def __call__(self, state):
        metric = self.metric(state)

        if self.best_score is None:
            self.best_score = metric
        elif self.comparison_function(metric, self.best_score):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = metric
            self.counter = 0

        return False


