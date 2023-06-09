import json
import typing
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, TypeAlias

from .entry import Entry, VerticalEntries, transpose_entries

Word: TypeAlias = Literal[
    "apple",
    "arm",
    "bank",
    "bass",
    "bow",
    "chair",
    "club",
    "crane",
    "deck",
    "digit",
    "hood",
    "java",
    "mole",
    "pitcher",
    "pound",
    "seal",
    "spring",
    "square",
    "trunk",
    "yard",
]
Split: TypeAlias = Literal["train", "test"]

WORDS: tuple = typing.get_args(Word)
DATA_ROOT: Path = Path("data/CoarseWSD-20")
WORDNET_MAPPINGS_PATH = DATA_ROOT / "wn_mappings.tsv"
OUT_OF_DOMAIN_DATA_PATH = DATA_ROOT / "CoarseWSD-20.outofdomain.tsv"


class Variant(Enum):
    """An enum representing the different variants of the CoarseWSD-20 dataset."""

    REGULAR = "CoarseWSD-20/{word}"
    BALANCED = "CoarseWSD-20_balanced/{word}"
    ONE_SHOT_1 = "CoarseWSD-20_nshot/set1/{word}_1"
    ONE_SHOT_2 = "CoarseWSD-20_nshot/set2/{word}_1"
    ONE_SHOT_3 = "CoarseWSD-20_nshot/set3/{word}_1"
    THREE_SHOT_1 = "CoarseWSD-20_nshot/set1/{word}_3"
    THREE_SHOT_2 = "CoarseWSD-20_nshot/set2/{word}_3"
    THREE_SHOT_3 = "CoarseWSD-20_nshot/set3/{word}_3"
    TEN_SHOT_1 = "CoarseWSD-20_nshot/set1/{word}_10"
    TEN_SHOT_2 = "CoarseWSD-20_nshot/set2/{word}_10"
    TEN_SHOT_3 = "CoarseWSD-20_nshot/set3/{word}_10"
    THIRTY_SHOT_1 = "CoarseWSD-20_nshot/set1/{word}_30"
    THIRTY_SHOT_2 = "CoarseWSD-20_nshot/set2/{word}_30"
    THIRTY_SHOT_3 = "CoarseWSD-20_nshot/set3/{word}_30"
    RATIO_1PCT = "CoarseWSD-20_ratios/1pct/{word}"
    RATIO_5PCT = "CoarseWSD-20_ratios/5pct/{word}"
    RATIO_10PCT = "CoarseWSD-20_ratios/10pct/{word}"
    RATIO_25PCT = "CoarseWSD-20_ratios/25pct/{word}"
    RATIO_50PCT = "CoarseWSD-20_ratios/50pct/{word}"
    RATIO_100PCT = "CoarseWSD-20_ratios/100pct/{word}"

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
class OutOfDomainEntry:
    """A single entry in the CoarseWSD20 out-of-domain dataset."""

    tokens: list[str]
    target_word: str
    target_word_id: int
    target_class: str


class WordDataset:
    """A CoarseWSD20-variant dataset for a single word.

    Raises
    ------
    `ValueError`
        When the dataset for the given word and variant doesn't exist.
    """

    _CLASSES_FILE = "classes_map.txt"
    _ENTRIES_FILE_TEMPLATE = "{split}.data.txt"
    _LABELS_FILE_TEMPLATE = "{split}.gold.txt"

    def __init__(self, root: Path, variant: Variant, word: Word):
        self.variant: Variant = variant
        self.word: Word = word
        self.path: Path = self._path(root, variant, word)

        if not self.path.exists():
            raise ValueError(f"Can't find dataset '{word}' for variant {variant.value}")

        self.load()

    def load(self) -> None:
        """Loads the dataset from disk."""
        self.classes: dict[str, str] = self._load_classes()
        self.train: list[Entry] = self._load_data("train")
        self.test: list[Entry] = self._load_data("test")

    @staticmethod
    def exists(root: Path, variant: Variant, word: Word) -> bool:
        """Checks if the dataset for the given word and variant exists.

        Parameters
        ----------
        `variant : Variant`
            The variant to check for.
        `word : Word`
            The word to check for.

        Returns
        -------
        `bool`
            True if the dataset exists, False otherwise.
        """
        return WordDataset._path(root, variant, word).exists()

    @staticmethod
    def _path(root: Path, variant: Variant, word: Word) -> Path:
        return root / variant.for_word(word)

    def _load_classes(self) -> dict[str, str]:
        with open(self.path / self._CLASSES_FILE, "r", encoding="utf-8") as file:
            return json.load(file)

    def _load_data(self, split: Split) -> list[Entry]:
        inputs_path = self.path / (self._ENTRIES_FILE_TEMPLATE.format(split=split))
        labels_path = self.path / (self._LABELS_FILE_TEMPLATE.format(split=split))

        entries = []

        with (
            open(inputs_path, "r", encoding="utf-8") as inputs_file,
            open(labels_path, "r", encoding="utf-8") as labels_file,
        ):
            for input_line, label in zip(inputs_file, labels_file):
                target_index, tokens = input_line.split("\t")

                entries.append(
                    Entry(
                        tokens=tokens.split(),
                        target_word_id=int(target_index),
                        target_class=self.classes[label.strip()],
                        target_class_id=int(label),
                    )
                )

        return entries

    def tokens(self, split: Split) -> list[list[str]]:
        """Returns the tokens for the given split.

        Parameters
        ----------
        `split : Split`
            The split to get the tokens for.

        Returns
        -------
        `list[list[str]]`
            The tokens for the given split.
        """
        return [entry.tokens for entry in self.get_data_split(split)]

    def get_data_split(self, split) -> list[Entry]:
        """Returns the entries for the given split.

        Parameters
        ----------
        `split : Split`
            The split to get the entries for.

        Returns
        -------
        `list[Entry]`
            The entries for the given split.
        """
        return getattr(self, split)

    def all(self) -> list[Entry]:
        """Returns all data entries, where train and test sets are concatentated (in that
        order).
        """
        return self.train + self.test

    def vertical(self, split: Split | None = None) -> VerticalEntries:
        """Returns the entries for the given split, vertically."""
        return transpose_entries(self.get_data_split(split) if split else self.all())


Dataset: TypeAlias = dict[Word, WordDataset]


@dataclass
class WordNetMapping:
    """A mapping from CoarseWSD-20 senses to WordNet synsets."""

    word: str
    sense: str
    synset_offset: str
    synset: str
    sense_key: str


def load_dataset(variant: Variant, root: str | Path = DATA_ROOT) -> Dataset:
    """Loads the CoarseWSD20-variant datasets for all words.

    Parameters
    ----------
    `variant : Variant`
        The variant to load.
    `root : str | Path, optional`
        The path to load the data from, by default `DATA_ROOT`

    Returns
    -------
    `Dataset`
        A mapping from words to their datasets.
    """
    root = Path(root)
    return {
        word: WordDataset(root, variant, word)
        for word in WORDS
        if WordDataset.exists(root, variant, word)
    }


def load_wordnet_mappings(
    path: str | Path = WORDNET_MAPPINGS_PATH,
) -> dict[str, WordNetMapping]:
    """Loads the WordNet mappings from disk.

    Parameters
    ----------
    `path : str | Path, optional`
        The path to load the data from, by default `WORDNET_MAPPINGS_PATH`

    Returns
    -------
    `dict[str, WordNetMapping]`
        A mapping from CoarseWSD20 senses/classes to WordNet synsets.
    """
    with open(Path(path), "r", encoding="utf-8") as file:
        next(file)  # skip header
        return {
            mapping.sense: mapping
            for mapping in (WordNetMapping(*line.strip().split("\t")) for line in file)
        }


def load_out_of_domain_data(
    path: str | Path = OUT_OF_DOMAIN_DATA_PATH,
) -> list[OutOfDomainEntry]:
    """Loads the out-of-domain data from disk.

    Parameters
    ----------
    `path : str | Path, optional`
        The path to load the data from, by default `OUT_OF_DOMAIN_DATA_PATH`

    Returns
    -------
    `list[OutOfDomainEntry]`
        The out-of-domain data.
    """
    with open(Path(path), "r", encoding="utf-8") as file:
        entries = []
        for line in file:
            word, sense, index, tokens = line.strip().split("\t")
            entries.append(
                OutOfDomainEntry(
                    tokens=tokens.split(),
                    target_word=word,
                    target_word_id=int(index),
                    target_class=sense,
                )
            )

        return entries
