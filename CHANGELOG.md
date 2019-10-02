Changelog
=========
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

# Unreleased
- fix default_save_directory typo
- replace progressbar block character by '+'


# # [0.2.0] - 2019-09-20
## New
- Add `ModuleTrainer.evaluate` method
- Add CsvWriter to evaluate method
- Add `filename_transform_function` argument to `SaveBestCheckpointCallback`
- Metric step method now return the intermediate computed values
- Rename `CsvWriter`'s `extra` argument to `extra_data_function`
- `CsvWriter` can now be called with the `extra_data` extra argument to define the extra data that will be logged
- Add `ModuleTrainer.load` method

## Change
- Rename `dateset_loader` to `dataloader`


# [0.1.0] - 2019-09-16
## New
- `ModuleTrainer` object
- `EarlyStopping`: stop training after a configurable period of stagnation
- Checkpointing: save model and estimator at regular intervals
- CSV file writer to output logs
- Several metrics are available: all default PyTorch loss functions, Accuracy, MAE
- Progress bar from console
- SIGINT handling: handle CTRL-C
- Model's data type (float32, float64)
- Full use of Pytorch's Cuda support
