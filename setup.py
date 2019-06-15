from setuptools import setup
from pytorchtrainer import __version__


setup(name="pytorchtrainer",
      version=__version__,

      description='PyTorch module trainer',
      license='MIT',

      author='Benoit Martin',
      author_email='benoitmartin88.pro@gmail.com',

      python_requires='>=3.5',

      keywords='pytorch trainer',

      packages=['pytorchtrainer',
                'pytorchtrainer.callback',
                'pytorchtrainer.metric',
                'pytorchtrainer.stop_condition'],

      install_requires=["pytorch >= 1.0.1"],

      # tests_require=["unittest"],
      test_suite='test',
      )

