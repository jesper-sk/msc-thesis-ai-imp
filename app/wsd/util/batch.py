from functools import partial
from itertools import islice
from typing import Iterable


def batched(iterable: Iterable, batch_size: int) -> Iterable:
    """Batch data from the `iterable` into tuples of length `batch_size`.
    The last batch may be shorter than `batch_size`.

    Parameters
    ----------
    `iterable : Iterable`
        The iterable that is to be batched.

    `batch_size : int`
        The size of every batch. Must be higher than zero.

    Returns
    -------
    `Iterable`
        The batched iterable.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, batch_size)):
        yield batch


def batched_(batch_size: int):
    """Returns a function that batches data from an iterable into tuples of length
    `batch_size`. The last batch may be shorter than `batch_size`.

    Parameters
    ----------
    `batch_size : int`
        The size of every batch. Must be higher than zero.

    Returns
    -------
    `Iterable`
        The batched iterable.
    """
    return partial(batched, batch_size=batch_size)


def num_batches(num_samples: int, batch_size: int) -> int:
    """Calculates the number of batches that are needed to process `num_samples` with
    `batch_size` samples per batch.

    Parameters
    ----------
    `num_samples : int`
        The total number of samples that are to be processed.
    `batch_size : int`
        The number of samples per batch.

    Returns
    -------
    `int`
        The number of batches that are needed to process `num_samples` with
    """
    quotient, remainder = divmod(num_samples, batch_size)
    return quotient + (remainder > 0)
