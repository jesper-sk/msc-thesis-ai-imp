# Standard library
from abc import ABC, abstractmethod
from typing import Iterator, Sequence

# Third-party imports
import numpy as np
import torch
from tqdm import tqdm

# Local imports
from ..data import Entry
from ..data import coarsewsd20 as cwsd
from ..util.helpers import num_batches


class Vectoriser(ABC):
    @abstractmethod
    def __call__(
        self, data: Sequence[Entry], batch_size: int = 1
    ) -> Iterator[torch.Tensor]:
        """Vectorise the target words of a list of entries in batches.

        Parameters
        ----------
        `data : list[Entry]`
            The entries to vectorise.
        `batch_size : int`, optional
            The size of the batches to use, by default 1. If `len(data)` is not
            divisible by `batch_size`, the last batch will be smaller.

        Returns
        -------
        `Iterator[torch.Tensor]`
            An iterator over the batches of vectors. Every tensor will have the
            shape `(batch_size, embedding_size)`.
        """
        pass


def vectorise_coarsewsd20(
    vectoriser: Vectoriser, dataset: cwsd.Dataset, batch_size: int = 1
) -> Iterator[tuple[str, np.ndarray]]:
    """Generate contextual embeddings of the target words of a CoarseWSD-20
    dataset.

    Parameters
    ----------
    `vectoriser : Vectoriser`
        The vectoriser to use.
    `dataset : cwsd.Dataset`
        The CoarseWSD-compatible dataset to vectorise.
    `batch_size : int`, optional
        The size of the batches to use, by default 1. If `len(data)` is not
        divisible by `batch_size`, the last batch will be smaller.

    Returns
    -------
    `Iterator[tuple[str, np.ndarray]]`
        An iterator over the dataset key and the corresponding contextual
        embeddings.

        The key is of the form `"{word}.{split}"`, where `word` is
        the target word and `split` is either `"train"` or `"test"`.
    """
    for word in dataset.keys():
        for split in ("train", "test"):
            data = dataset[word].get_data_split(split)
            yield f"{word}.{split}", np.concatenate(
                tuple(
                    tqdm(
                        map(
                            lambda x: x.numpy(),
                            vectoriser(data, batch_size=batch_size),
                        ),
                        f"Vectorising {split} split of word '{word}'",
                        num_batches(len(data), batch_size),
                    )
                )
            )
