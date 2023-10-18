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
    1. The WSD Evaluation framework data can be downloaded from [here](http://lcl.uniroma1.it/wsdeval/). The default location to extrarct is `data/wsdeval/WSD_Evaluation_Framework`.
    1. The XL-WSD data can be downloaded from [here](https://sapienzanlp.github.io/xl-wsd/) ("Data"). The default location to extract is `data/xl-wsd`. 

3. Run the CLI of the project. This can be done through pipenv.
    ```sh
    pipenv run python app
    ```
## Structure
Implementational code can be found in three places. Firstly, there is the CLI, which can be run through the `pipenv run python app` command. The following functions are available:
- `vectorise` is used to create CWEs from target sentences. 
- `prep-ewiser` prepares the CoarseWSD-20 dataset to be interpreted by the EWISER model.
- `split-embeddings` splits the CoarseWSD-20 embeddings into synset state clouds using ground truth.
- `lemmas` extracts all lemmata from a WSD Evaluation Framework variant and saves them as a csv.
- `filter-bookcorpus` filters the original BookCorpus set of sentences into those that use a given set of nouns.

Secondly, there are a number of interactive scripts in the `app/` directory. These are designed to work in your editor's scientific mode, using "`% ##`" to specify individual code blocks. This should work at least in VSCode and JetBrains PyCharm. 

Thirdly, the EWISER submodule (`repos/ewiser`) has some interfacing scripts in the `bin` directory. These are used to interface with the EWISER implementation (which was not made by me, see to the [submodule's README](https://github.com/jesper-sk/ewiser)).

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

## Resources
- [BookCorpus sentences sorted by synset](https://github.com/jesper-sk/msc-thesis-bookcorpus-synset-sentences)
- [EWISER fork used in this repository](https://github.com/jesper-sk/ewiser)
- [CoarseWSD fork used in this repository](https://github.com/jesper-sk/coarsewsd-20)