#!/usr/bin/env python3

from argparse import ArgumentParser


def run(args):
    raise NotImplementedError()     # TODO


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--val_batch_size', type=int, default=1000,
                        help='input batch size for validation (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()

    run(args)
