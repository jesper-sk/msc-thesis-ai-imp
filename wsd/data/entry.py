from dataclasses import dataclass


@dataclass
class Entry:
    """A single data entry."""

    tokens: list[str]
    target_word_id: int
    target_class: str
    target_class_id: int

    def __iter__(self):
        return iter(
            (
                self.tokens,
                self.target_word_id,
                self.target_class,
                self.target_class_id,
            )
        )

    def __len__(self):
        return 4
