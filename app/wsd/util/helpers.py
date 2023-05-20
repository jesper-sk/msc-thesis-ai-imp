from typing import Iterable
from itertools import islice


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
