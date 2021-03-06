#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This is a simple MNIST example.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import pytorchtrainer as ptt


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':

    model = Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.NLLLoss()
    batch_size = 128

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)

    validation_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True)

    trainer = ptt.create_default_trainer(model, optimizer, criterion, verbose=1)

    # Validate after every 200 iteration and after every epoch
    trainer.register_post_iteration_callback(ptt.callback.ValidationCallback(validation_loader, metric=ptt.metric.TorchLoss(criterion)), frequency=200)

    validation_callback = ptt.callback.ValidationCallback(validation_loader, metric=ptt.metric.TorchLoss(criterion))
    trainer.register_post_epoch_callback(validation_callback, frequency=1)

    # compute accuracy
    accuracy_callback = ptt.callback.MetricCallback(metric=ptt.metric.Accuracy(prediction_transform=lambda x: x.argmax(dim=1, keepdim=False)))
    trainer.register_post_iteration_callback(accuracy_callback, frequency=1)

    # add validation loss and accuracy in progress bar
    trainer.add_progressbar_metric("validation loss %.4f | accuracy %.2f", [validation_callback, accuracy_callback])

    trainer.train(train_loader,
                  max_epochs=10,
                  stop_condition=ptt.stop_condition.EarlyStopping(patience=2,
                                                                  metric=lambda state: getattr(state, accuracy_callback.state_attribute_name),
                                                                  comparison_function=lambda metric, best: round(metric, 2) <= round(best, 2)))
