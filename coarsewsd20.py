import typing

from enum import Enum
from pathlib import Path
from typing import Dict, Literal, Tuple

Word = Literal['apple', 'arm', 'bank', 'bass', 'bow', 'chair', 'club',
               'crane', 'deck', 'digit', 'hood', 'java', 'mole',
               'pitcher', 'pound', 'seal', 'spring', 'square', 'trunk',
               'yard']
Split = Literal['train', 'test']

DATA_PATH: Path = Path('data/CoarseWSD-20')
WORDS: Tuple = typing.get_args(Word)


class Dataset(Enum):
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


class WordDataset:
    classes: Dict[str, str] = None

    _path: Path = None

    def __init__(self, dataset: Dataset, word: Word):
        pass

    @staticmethod
    def _mk_path(dataset: Dataset, word: Word):
        return DATA_PATH / (dataset.value % word)
