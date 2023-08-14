from dataclasses import dataclass
from typing import Iterable

from .tokens import TokenInput, TokenInputConvertible


@dataclass
class Entry(TokenInputConvertible):
    """A single data entry."""

    tokens: list[str]
    target_word_id: int
    target_class: str
    target_class_id: int

    def unpack(self) -> tuple[list[str], int, str, int]:
        return (
            self.tokens,
            self.target_word_id,
            self.target_class,
            self.target_class_id,
        )

    def to_tokens(self) -> TokenInput:
        return TokenInput(self.tokens, self.target_word_id)


@dataclass
class VerticalEntries:
    """A collection of entries, stored vertically."""

    tokens: tuple[list[str]]
    target_word_ids: tuple[int]
    target_classes: tuple[str]
    target_class_ids: tuple[int]

    @classmethod
    def make(cls, entries: Iterable[Entry]):
        """Transpose a collection of entries. Instead of a list of entries, this will return a
        single data structure with lists of the corresponding fields.

        Parameters
        ----------
        entries : Iterable[Entry]
            The entries to transpose.

        Returns
        -------
        VerticalEntries
            The transposed entries.
        """
        return cls(*zip(*map(lambda x: x.unpack(), entries)))  # type: ignore


def transpose_entries(entries: Iterable[Entry]):
    """Transpose a collection of entries. Instead of a list of entries, this will return a
    single data structure with lists of the corresponding fields.

    Parameters
    ----------
    entries : Iterable[Entry]
        The entries to transpose.

    Returns
    -------
    VerticalEntries
        The transposed entries.
    """
    return VerticalEntries(*zip(*map(lambda x: x.unpack(), entries)))  # type: ignore
