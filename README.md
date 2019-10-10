# PyTorch trainer

[![CircleCI](https://circleci.com/gh/benoitmartin88/pytorchtrainer/tree/master.svg?style=svg)](https://circleci.com/gh/benoitmartin88/pytorchtrainer/tree/master)

Are you tired of writing those same epoch and data-loader loops to train your PyTorch module ?
Look no further, PyTorch trainer is a library that hides all those boring training lines of code that should be native to PyTorch. 

You will also benefit from the following features:

- Early stopping: stop training after a period of stagnation
- Checkpointing: save model and estimator at regular intervals
- CSV file writer to output logs
- Several metrics are available: all default PyTorch loss functions, Accuracy, MAE
- Progress bar from console
- SIGINT handling: handle CTRL-C
- Model's data type (`float32`, `float64`) 


## Example

Code examples can be found in the [example folder](https://github.com/benoitmartin88/pytorchtrainer/tree/master/examples).

Here is a simple example:

``` python

import torch
import pytorchtrainer as ptt


# Your usual model, optimizer, loss function and data loaders
model = MyModel()
optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()
train_loader = MyTrainDataloader()
validation_loader = MyValidationDataloader()


# instantiate a default trainer
trainer = ptt.create_default_trainer(model, optimizer, criterion)

# optionally save a checkpoint after every 10 epochs
trainer.register_post_epoch_callback(ptt.checkpoint.SaveCheckpointCallback(save_every=10))

# optionally compute validation loss after every epoch
validation_callback = ptt.callback.ValidationCallback(validation_loader, ptt.metric.TorchLoss(criterion), validate_every=1)
trainer.register_post_epoch_callback(validation_callback)

# optionally save training and validation loss after every iteration using default save directory
trainer.register_post_iteration_callback(ptt.callback.CsvWriter(save_every=1,
                                                                extra_header=[validation_callback.state_attribute_name],
                                                                extra_data_function=lambda state: [state.get(validation_callback.state_attribute_name)]))
# run the training
trainer.train(train_loader, max_epochs=100)

```

## Dependencies

- python > 3.5
- pytorch > 1.0.0 (install instructions from the official [PyTorch website](https://pytorch.org/get-started/locally))


## Contributing

Feel free to submit an issue or pull request. But before you do please read the [contributing guidelines](CONTRIBUTING.md)

