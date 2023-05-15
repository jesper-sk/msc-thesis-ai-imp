[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Implementation for MSc thesis AI

## Installing the project
1. Clone the repository. Make sure to clone the submodule(s) as well by using the `--recurse-submodules` flag. You can also clone the submodule(s) afterwards by using `git submodule update --init --recursive`.
2. Install the dependencies through by running `pipenv install` in the root of the project. You can install pipenv if needed by running `pip install pipenv`.
3. You can now run the CLI by running `pipenv run python wsd` in the root of the 
project.

## Static analysers in use
1. Black code formatter
2. Mypy static type analyser
3. Flake8 linter
4. _(optional)_ Pylint linter