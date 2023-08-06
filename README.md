[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Implementation for MSc thesis AI

## Installing the project
1. Clone the repository. Make sure to clone the submodule(s) as well by using the `--recurse-submodules` flag. You can also clone the submodule(s) afterwards by using: 
    ```sh
    git submodule update --init
    ```

2. Install the project's dependencies. This project uses `pipenv` for dependency manegement.
    ```sh
    pipenv install --dev
    ```
    1. You can install pipenv if needed by running `pip3 install pipenv` (`pip` on Windows)
    1. The `--dev` flag makes sure all development dependencies are also installed. If you are just planning to run the project and not develop from it, you can omit it. 
    1. You can also install all dependencies through `pip` directly, using the provided `requirements.txt` file. It is recommended to use a virtual environment nonetheless.

3. Ensure the necessary project resources are installed.
    1. You can download WordNet 3.0 [here](http://wordnetcode.princeton.edu/3.0/WordNet-3.0.tar.gz). The default location to extract is `data/WordNet-3.0`.
    1. The WSD Evaluation framework data can be downloaded from [here](http://lcl.uniroma1.it/wsdeval/). The default location to extrarct is `data/WSD_Evaluation_Framework`.
    1. The XL-WSD data can be downloaded from [here](https://sapienzanlp.github.io/xl-wsd/) ("Data"). The default location to extract is `data/xl-wsd`. 

3. Run the CLI of the project. This can be done through pipenv.
    ```sh
    pipenv run python app
    ```

## Running unit tests
Run: 
```sh
pipenv run python -m unittest discover tests "*.py"
```

## Static analysers in use
1. Black code formatter
2. Mypy static type analyser
3. Flake8 linter
4. _(optional)_ Pylint linter