import torch


def batch_to_tensor(batch, device=None, non_blocking=False):
    x, y = batch
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)

    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    x = x.to(device=device, non_blocking=non_blocking)
    y = y.to(device=device, non_blocking=non_blocking)
    return x, y


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    from: https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a

    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    import sys

    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()
