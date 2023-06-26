# %%
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy._typing as tp
import numpy.linalg as la
from numpy._typing import ArrayLike, NDArray

ArrayLikeFloat = tp._ArrayLikeFloat_co


def is_square_matrix(arr: NDArray[Any]):
    shape = arr.shape
    return len(shape) == 2 and shape[0] == shape[1]


def eye_like(arr: np.ndarray, dtype=None) -> np.ndarray:
    return np.eye(arr.shape[0], dtype=dtype)


def cosine_similarity(a: ArrayLike, b: ArrayLike) -> float:
    return np.dot(a, b) / (la.norm(a) * la.norm(b))


def angle_between_vectors(a: ArrayLike, b: ArrayLike) -> float:
    return np.arccos(cosine_similarity(a, b))


def lissajous(a: float, b: float, delta_f: float, range: ArrayLike):
    range = np.asarray(range)
    x = np.sin(a * range + (np.pi / delta_f))
    y = np.sin(b * range)
    return np.hstack((x[:, None], y[:, None]))


@dataclass
class Ellipse:
    semiaxis_x: float
    semiaxis_y: float
    angle: float


@dataclass
class Ellipsoid:
    semiaxis_x: float
    semiaxis_y: float
    semiaxis_z: float
    polar_angle: float
    azimuthal_angle: float


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

        aperture_matrix = (aperture ** (-2)) * eye_like(correlation_matrix)

        conceptor_matrix = (
            la.inv(correlation_matrix + aperture_matrix) @ correlation_matrix + 1e-10
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

    @property
    def inv(self) -> Conceptor:
        return la.inv(self)  # type: ignore

    def neg(self) -> Conceptor:
        return eye_like(self) - self  # type: ignore

    def con(self, other: Conceptor) -> Conceptor:
        return (self.inv + other.inv + eye_like(self)).inv  # type: ignore

    def dis(self, other: Conceptor) -> Conceptor:
        id = eye_like(self)
        return (
            id + (self @ (id - self).inv + other @ (id - other).inv).inv  # type: ignore
        ).inv

    def to_ellipse_params(self) -> tuple[float, float, float]:
        """Returns ellipse parameters (a, b, theta) for the conceptor. Note that the
        conceptor must be of order 2.

        Returns
        -------
        tuple[float, float, float]
            Ellipse parameters (a, b, theta) for the conceptor.

        Explanation
        -----------
        The principal axes of the ellipse are the eigenvectors of the conceptor. The
        lengths of the principal axes are the square roots of the eigenvalues of the
        conceptor. The angle of rotation of the ellipse is the angle between the first
        principal axis and the x-axis.

        References
        ----------
        - [1] https://en.wikipedia.org/wiki/Ellipse#General_ellipse
        - [2] https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors
        - [3] https://math.stackexchange.com/questions/612981/find-angle-at-given-points-in-ellipse
        """
        assert self.order == 2

        s = la.svd(self)[1]
        return s[0], s[1], np.arctan2(self[1, 0], self[0, 0])

    def to_ellipse2(self):
        assert self.order == 2

        u, s, _ = la.svd(self)
        width, height = s * 2
        angle = cosine_similarity(u @ [1, 0], [1, 0])
        return width, height, angle

    def to_ellipse(self) -> Ellipse:
        """https://en.wikipedia.org/wiki/Ellipsoid#As_a_quadric"""
        assert self.order == 2

        eigenvalues, eigenvectors = la.eig(self)
        semiaxes = tuple(eigenvalues ** (-0.5))

        ellipse_principal_axis_x = eigenvectors[:, 0]
        unit_axis_x = np.array([0, 1])

        angle_rad = angle_between_vectors(ellipse_principal_axis_x, unit_axis_x)

        return Ellipse(semiaxes[0], semiaxes[1], angle_rad)


# %%
