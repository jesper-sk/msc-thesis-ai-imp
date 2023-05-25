# Standard library
from abc import ABC, abstractmethod
from typing import Iterable, Iterator, Sequence

# Third-party imports
import numpy as np
import torch
from tqdm import tqdm

# Local imports
from ..data import Entry
from ..data import coarsewsd20 as cwsd
from ..util.batch import batched, num_batches


class Vectoriser(ABC):
    @abstractmethod
    def __call__(self, batch: Iterable[Entry]) -> torch.Tensor:
        pass

    @abstractmethod
    def load_prepare_models(self) -> None:
        pass

    def call(
        self, data: Sequence[Entry], batch_size: int = 1
    ) -> Iterator[torch.Tensor]:
        """Vectorise the given data and return the output of the model for each sentence.

        Parameters
        ----------
        `data : Sequence[Entry]`
            The data to vectorise.
        `batch_size : int, optional`
            The batch size to use, by default 1

        Yields
        ------
        `Iterator[Tensor]`
            The output of the model for each sentence in the batch.
        """
        self.load_prepare_models()

        for batch in batched(data, batch_size):
            yield self(batch)


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
                            vectoriser.call(data, batch_size=batch_size),
                        ),
                        f"Vectorising {split} split of word '{word}'",
                        num_batches(len(data), batch_size),
                    )
                )
            )
