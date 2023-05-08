import json
import typing
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Tuple

Word = Literal['apple', 'arm', 'bank', 'bass', 'bow', 'chair', 'club',
               'crane', 'deck', 'digit', 'hood', 'java', 'mole',
               'pitcher', 'pound', 'seal', 'spring', 'square', 'trunk',
               'yard']
Split = Literal['train', 'test']

DATA_ROOT: Path = Path('data/CoarseWSD-20')
WORDS: Tuple = typing.get_args(Word)


class Variant(Enum):
    """An enum representing the different variants of the CoarseWSD-20 dataset.
    """

    REGULAR = 'CoarseWSD-20/{word}'
    BALANCED = 'CoarseWSD-20_balanced/{word}'
    ONE_SHOT_1 = 'CoarseWSD-20_nshot/set1/{word}_1'
    ONE_SHOT_2 = 'CoarseWSD-20_nshot/set2/{word}_1'
    ONE_SHOT_3 = 'CoarseWSD-20_nshot/set3/{word}_1'
    THREE_SHOT_1 = 'CoarseWSD-20_nshot/set1/{word}_3'
    THREE_SHOT_2 = 'CoarseWSD-20_nshot/set2/{word}_3'
    THREE_SHOT_3 = 'CoarseWSD-20_nshot/set3/{word}_3'
    TEN_SHOT_1 = 'CoarseWSD-20_nshot/set1/{word}_10'
    TEN_SHOT_2 = 'CoarseWSD-20_nshot/set2/{word}_10'
    TEN_SHOT_3 = 'CoarseWSD-20_nshot/set3/{word}_10'
    THIRTY_SHOT_1 = 'CoarseWSD-20_nshot/set1/{word}_30'
    THIRTY_SHOT_2 = 'CoarseWSD-20_nshot/set2/{word}_30'
    THIRTY_SHOT_3 = 'CoarseWSD-20_nshot/set3/{word}_30'
    RATIO_1PCT = 'CoarseWSD-20_ratios/1pct/{word}'
    RATIO_5PCT = 'CoarseWSD-20_ratios/5pct/{word}'
    RATIO_10PCT = 'CoarseWSD-20_ratios/10pct/{word}'
    RATIO_25PCT = 'CoarseWSD-20_ratios/25pct/{word}'
    RATIO_50PCT = 'CoarseWSD-20_ratios/50pct/{word}'
    RATIO_100PCT = 'CoarseWSD-20_ratios/100pct/{word}'

    def for_word(self, word: Word) -> str:
        """Returns the path for the given word.

        Parameters
        ----------
        word : Word
            The word to get the path for.

        Returns
        -------
        str
            The partial path for the given word.
        """
        return self.value.format(word=word)


@dataclass
class Entry:
    """A single entry in a CoarseWSD20-variant dataset.
    """
    tokens: List[str]
    target_index: int
    target_class: str
    target_class_index: int


class WordDataset:
    """A CoarseWSD20-variant dataset for a single word. 

    Raises
    ------
    ValueError
        When the dataset for the given word and variant doesn't exist.
    """
    _CLASSES_FILE = 'classes_map.txt'
    _ENTRIES_FILE_TEMPLATE = '{split}.data.txt'
    _LABELS_FILE_TEMPLATE = '{split}.gold.txt'

    path: Path = None
    classes: Dict[str, str] = None
    train: List[Entry] = None
    test: List[Entry] = None

    def __init__(self, variant: Variant, word: Word):
        self.path = self._path(variant, word)

        if not self.path.exists():
            raise ValueError(
                f"Can't find dataset '{word}' for variant {variant.value}"
            )

        self.load()

    def load(self):
        """Loads the dataset from disk.
        """
        self.classes = self._load_classes()
        self.train = self._load_data('train')
        self.test = self._load_data('test')

    @staticmethod
    def exists(variant: Variant, word: Word) -> bool:
        """Checks if the dataset for the given word and variant exists.

        Parameters
        ----------
        variant : Variant
            The variant to check for.
        word : Word
            The word to check for.

        Returns
        -------
        bool
            True if the dataset exists, False otherwise.
        """
        return WordDataset._path(variant, word).exists()

    @staticmethod
    def _path(variant: Variant, word: Word) -> Path:
        return DATA_ROOT / variant.for_word(word)

    def _load_classes(self) -> Dict[str, str]:
        with open(self.path / self._CLASSES_FILE, 'r', encoding='utf-8') as file:
            return json.load(file)

    def _load_data(self, split: Split) -> List[Entry]:
        inputs_path = self.path / \
            (self._ENTRIES_FILE_TEMPLATE.format(split=split))
        labels_path = self.path / \
            (self._LABELS_FILE_TEMPLATE.format(split=split))

        entries = []

        with open(inputs_path, 'r', encoding='utf-8') as inputs_file, \
                open(labels_path, 'r', encoding='utf-8') as labels_file:

            for input_line, label in zip(inputs_file, labels_file):
                target_index, tokens = input_line.split('\t')
                target_index = int(target_index)
                tokens = tokens.split()
                target_class = self.classes[label.strip()]
                target_class_index = int(label)

                entries.append(
                    Entry(
                        tokens, target_index, target_class,
                        target_class_index
                    )
                )

        return entries


class FullDataset(Mapping[Word, WordDataset]):
    """A CoarseWSD20-variant dataset for all words.
    """
    _data: Dict[Word, WordDataset] = None

    def __init__(self, variant: Variant):
        self._data = {
            word: WordDataset(variant, word) for word in WORDS
            if WordDataset.exists(variant, word)
        }

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key: Word) -> WordDataset:
        return self._data[key]

    def __iter__(self) -> Any:
        return iter(self._data)
