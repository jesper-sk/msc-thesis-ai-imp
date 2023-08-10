from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable


@dataclass
class TokenInput:
    tokens: list[str]
    target_token_id: int


class TokenInputConvertible(ABC):
    @abstractmethod
    def to_tokens(self) -> TokenInput:
        """Convert the given class to a TokenInput."""
        pass


@dataclass
class TokenBatch:
    tokens: tuple[list[str]]
    target_token_id: tuple[int]

    @classmethod
    def make(cls, tokens: Iterable[TokenInput]):
        return cls(*zip(*map(unpack_tokens, tokens)))  # type: ignore


def unpack_tokens(input: TokenInput | TokenInputConvertible) -> tuple[list[str], int]:
    if isinstance(input, TokenInput):
        return (input.tokens, input.target_token_id)
    return unpack_tokens(input.to_tokens())
