# %%
from __future__ import annotations

from typing import Any

import numpy as np
import numpy._typing as tp
import numpy.linalg as la
from numpy._typing import NDArray

ArrayLikeFloat = tp._ArrayLikeFloat_co


def is_square_matrix(arr: NDArray[Any]):
    shape = arr.shape
    return len(shape) == 2 and shape[0] == shape[1]


def eye_like(conceptor: Conceptor, dtype=None) -> np.ndarray:
    return np.eye(conceptor.order, dtype=dtype)


class Conceptor(np.ndarray):
    @staticmethod
    def from_state_matrix(matrix: ArrayLikeFloat, aperture: float = 20):
        correlation_matrix = np.corrcoef(np.asarray(matrix).T)

        return Conceptor.from_correlation_matrix(correlation_matrix, aperture)

    @staticmethod
    def from_correlation_matrix(
        correlation_matrix: NDArray[Any], aperture: float = 0.1
    ):
        correlation_matrix = np.asarray(correlation_matrix)
        assert is_square_matrix(correlation_matrix)
        conceptor_matrix = (
            la.inv(
                correlation_matrix
                + aperture ** (-2) * np.eye(correlation_matrix.shape[0])
            )
            @ correlation_matrix
            + 1e-10
        )

        return Conceptor(conceptor_matrix, aperture)

    def __new__(
        cls,
        conceptor_matrix,
        aperture: float | None = None,
    ):
        obj = np.asarray(conceptor_matrix).view(cls)  # Calls __array_finalize__
        obj.aperture = aperture

        return obj

    def __array_finalize__(self, obj: NDArray[Any] | None) -> None:
        if obj is None:
            # There's nothing to finalize
            return

        if (obj_order := getattr(obj, "order", None)) is not None:
            # We're in new-from-template context, we can take attributes from template
            self.order = obj_order
            self.aperture = getattr(obj, "aperture")
        else:
            # We're in view casting context, we can infer order from shape
            assert is_square_matrix(obj)
            self.order = obj.shape[0]

    def inv(self) -> Conceptor:
        return la.inv(self)  # type: ignore

    def logic_not(self) -> Conceptor:
        return eye_like(self) - self  # type: ignore

    def logic_and(self, other: Conceptor) -> Conceptor:
        return (self.inv() + other.inv() + eye_like(self)).inv()  # type: ignore

    def logic_or(self, other: Conceptor) -> Conceptor:
        id = eye_like(self)
        return (
            id + (self @ (id - self).inv() + other @ (id - other).inv()).inv()  # type: ignore
        ).inv()


# %%
