import numpy as np
import numpy.linalg as la

from .conceptor import Conceptor


def is_positive_definite(c: Conceptor) -> np.bool_:
    return np.all(la.eigvals(c) > 0)
