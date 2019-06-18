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
    trainer.register_post_iteration_callback(ptt.callback.ValidationCallback(validation_loader, metric=ptt.metric.Loss(criterion), validate_every=200))

    validation = ptt.callback.ValidationCallback(validation_loader, metric=ptt.metric.Loss(criterion), validate_every=1)
    trainer.register_post_epoch_callback(validation)

    # compute accuracy
    accuracy = ptt.callback.MetricCallback(
        metric=ptt.metric.Accuracy(prediction_transform=lambda x: x.argmax(dim=1, keepdim=False)),
        frequency=1)
    trainer.register_post_iteration_callback(accuracy)

    # add validation loss and accuracy in progress bar
    trainer.add_progressbar_metric("validation loss %.4f | accuracy %.2f", [validation.state_attribute_name, accuracy.state_attribute_name])

    trainer.train(train_loader, max_epochs=10)
