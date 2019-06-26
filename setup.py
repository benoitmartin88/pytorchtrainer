from setuptools import setup
import os


def find_version(*file_paths):
    def read(*parts):
        here = os.path.abspath(os.path.dirname(__file__))
        with open(os.path.join(here, *parts)) as fp:
            return fp.read().strip()

    import re
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def readme():
    """print long description"""
    with open('README.rst') as f:
        return f.read()


version = find_version("pytorchtrainer", "__init__.py")


setup(name="pytorchtrainer",
      version=version,

      description='PyTorch module trainer',
      long_description=readme(),
      license='MIT',

      url='https://github.com/benoitmartin88/pytorchtrainer',
      project_urls={
          'Bug Reports': 'https://github.com/benoitmartin88/pytorchtrainer/issues',
          'Source': 'https://github.com/benoitmartin88/pytorchtrainer',
      },

      author='Benoit Martin',
      author_email='benoitmartin88.pro@gmail.com',

      python_requires='>=3.5',

      keywords='pytorch trainer',

      packages=['pytorchtrainer',
                'pytorchtrainer.callback',
                'pytorchtrainer.metric',
                'pytorchtrainer.stop_condition'],

      install_requires=["torch >= 1.0.0"],

      # tests_require=["unittest"],
      test_suite='test',

      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],
      )

