# Future
from __future__ import annotations

# Standard library
import itertools as it
from typing import Any, Callable, Iterable


class Pipeline(Iterable):
    def __init__(self, source: Iterable):
        self.source: Iterable = source
        self.spout: Iterable = source

    def __iter__(self):
        return iter(self.spout)

    def __or__(self, other: Callable[[Iterable], Iterable]) -> Pipeline:
        return self.chain(other)

    def __add__(self, other: Iterable) -> Pipeline:
        return self.combine(other)

    def __mul__(self, other: Iterable) -> Pipeline:
        return self.product(other)

    def __gt__(self, reducer: Callable[[Iterable], Any]) -> Any:
        return self.into(reducer)

    def combine(self, iterable: Iterable) -> Pipeline:
        self.spout = zip(self.spout, iterable)
        return self

    def product(self, iterable: Iterable) -> Pipeline:
        self.spout = it.product(self.spout, iterable)
        return self

    def chain(self, iter_factory: Callable[..., Iterable], *args, **kwargs) -> Pipeline:
        self.spout = iter_factory(self.spout, *args, **kwargs)
        return self

    def with_chain(
        self, iter_factory: Callable[..., Iterable], *args, **kwargs
    ) -> Pipeline:
        self.spout = zip(self.spout, iter_factory(self.spout, *args, **kwargs))
        return self

    def map(self, func: Callable) -> Pipeline:
        self.spout = map(func, self.spout)
        return self

    def starmap(self, func: Callable) -> Pipeline:
        self.spout = it.starmap(func, self.spout)
        return self

    def filter(self, func: Callable[..., bool]) -> Pipeline:
        self.spout = filter(func, self.spout)
        return self

    def starfilter(self, func: Callable[..., bool]) -> Pipeline:
        self.spout = it.compress(self.spout, it.starmap(func, self.spout))
        return self

    def with_map(self, func: Callable) -> Pipeline:
        self.spout = zip(self, map(func, self.spout))
        return self

    def with_filter(self, func: Callable[..., bool]) -> Pipeline:
        self.spout = zip(self, filter(func, self.spout))
        return self

    def into(self, reducer: Callable[[Iterable], Any]) -> Any:
        return reducer(self.spout)


def pipe(iterable: Iterable) -> Pipeline:
    return Pipeline(iterable)
