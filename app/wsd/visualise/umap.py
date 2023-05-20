# Third-party imports
import numpy as np
from umap import UMAP


def umap(input: np.ndarray) -> np.ndarray:
    return UMAP().fit_transform(input)
