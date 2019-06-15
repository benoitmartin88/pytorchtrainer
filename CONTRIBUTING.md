# Contributing guidelines

## Pull Request Checklist

Before sending your pull requests, make sure you followed this list.

- Read [contributing guidelines](CONTRIBUTING.md).

## How to become a contributor and submit your own code


### Contributing code

If you want to contribute, start working through the codebase,
navigate to the
[Github "issues" tab](https://github.com/benoitmartin88/pytorch-trainer/issues) and start
looking through interesting issues. 


### Contribution guidelines and standards

Before sending your pull request for
[review](https://github.com/benoitmartin88/pytorch-trainer/pulls),
make sure your changes are consistent with the guidelines and follow the
TensorFlow coding style.

#### General guidelines and philosophy for contribution

*   Include unit tests when you contribute new features, as they help to a)
    prove that your code works correctly, and b) guard against future breaking
    changes to lower the maintenance cost.
*   Bug fixes also generally require unit tests, because the presence of bugs
    usually indicates insufficient test coverage.
*   Keep API compatibility in mind


#### Writing and running unit-tests

Unit tests are stored in the `test` directory at the repository root.
To leverage the maximum out of python's unittest module, test files should be prefixed 
by `test_` in order to comply with unittest's `test_*.py` pattern.


All the unit tests can be run using the following command from the repository root: 
`PYTHONPATH=pytorch-trainer python -m unittest discover -s test/`
