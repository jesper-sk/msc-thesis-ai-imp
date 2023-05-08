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
    REGULAR = 'CoarseWSD-20/%s'
    BALANCED = 'CoarseWSD-20_balanced/%s'
    ONE_SHOT_1 = 'CoarseWSD-20_nshot/set1/%s_1'
    ONE_SHOT_2 = 'CoarseWSD-20_nshot/set2/%s_1'
    ONE_SHOT_3 = 'CoarseWSD-20_nshot/set3/%s_1'
    THREE_SHOT_1 = 'CoarseWSD-20_nshot/set1/%s_3'
    THREE_SHOT_2 = 'CoarseWSD-20_nshot/set2/%s_3'
    THREE_SHOT_3 = 'CoarseWSD-20_nshot/set3/%s_3'
    TEN_SHOT_1 = 'CoarseWSD-20_nshot/set1/%s_10'
    TEN_SHOT_2 = 'CoarseWSD-20_nshot/set2/%s_10'
    TEN_SHOT_3 = 'CoarseWSD-20_nshot/set3/%s_10'
    THIRTY_SHOT_1 = 'CoarseWSD-20_nshot/set1/%s_30'
    THIRTY_SHOT_2 = 'CoarseWSD-20_nshot/set2/%s_30'
    THIRTY_SHOT_3 = 'CoarseWSD-20_nshot/set3/%s_30'
    RATIO_1PCT = 'CoarseWSD-20_ratios/1pct/%s'
    RATIO_5PCT = 'CoarseWSD-20_ratios/5pct/%s'
    RATIO_10PCT = 'CoarseWSD-20_ratios/10pct/%s'
    RATIO_25PCT = 'CoarseWSD-20_ratios/25pct/%s'
    RATIO_50PCT = 'CoarseWSD-20_ratios/50pct/%s'
    RATIO_100PCT = 'CoarseWSD-20_ratios/100pct/%s'

    def for_word(self, word: Word) -> str:
        return self.value % word


@dataclass
class Entry:
    tokens: List[str]
    target_index: int
    target_class: str
    target_class_index: int


class WordDataset:
    _CLASSES_FILE = 'classes_map.txt'
    _ENTRIES_FILE_TEMPLATE = '%s.data.txt'
    _LABELS_FILE_TEMPLATE = '%s.gold.txt'

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

    @staticmethod
    def _path(variant: Variant, word: Word) -> Path:
        return DATA_ROOT / variant.for_word(word)

    def load(self):
        self.classes = self._load_classes()
        self.train = self._load_data('train')
        self.test = self._load_data('test')

    def _load_classes(self) -> Dict[str, str]:
        with open(self.path / self._CLASSES_FILE, 'r', encoding='utf-8') as file:
            return json.load(file)

    def _load_data(self, split: Split) -> List[Entry]:
        inputs_path = self.path / (self._ENTRIES_FILE_TEMPLATE % split)
        labels_path = self.path / (self._LABELS_FILE_TEMPLATE % split)

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

    @staticmethod
    def exists(variant: Variant, word: Word):
        return WordDataset._path(variant, word).exists()


class FullDataset(Mapping[Word, WordDataset]):
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
