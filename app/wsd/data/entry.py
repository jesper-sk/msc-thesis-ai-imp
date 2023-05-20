# Standard library
from dataclasses import dataclass
from typing import Iterable


@dataclass
class Entry:
    """A single data entry."""

    tokens: list[str]
    target_word_id: int
    target_class: str
    target_class_id: int

    def unpack(self) -> tuple[list[str], int, str, int]:
        return (
            self.tokens,
            self.target_class_id,
            self.target_class,
            self.target_class_id,
        )


@dataclass
class VerticalEntries:
    """A collection of entries, stored vertically."""

    tokens: tuple[list[str]]
    target_word_ids: tuple[int]
    target_classes: tuple[str]
    target_class_ids: tuple[int]


def transpose_entries(
    entries: Iterable[Entry],
) -> VerticalEntries:
    return VerticalEntries(*zip(*map(lambda x: x.unpack(), entries)))
