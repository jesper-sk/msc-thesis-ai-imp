# Standard library
from abc import ABC, abstractmethod
from typing import Iterator

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
        self, data: list[Entry], batch_size: int = 1
    ) -> Iterator[torch.Tensor]:
        pass


def vectorise_coarsewsd20(
    vectoriser: Vectoriser, dataset: cwsd.Dataset, batch_size: int = 1
) -> Iterator[tuple[str, np.ndarray]]:
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
