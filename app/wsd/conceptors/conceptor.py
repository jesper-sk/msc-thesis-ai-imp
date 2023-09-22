# %%
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np
import numpy._typing as tp
import numpy.linalg as la
from matplotlib.patches import Ellipse as EllipsePatch
from numpy._typing import ArrayLike, NDArray

ArrayLikeFloat = tp._ArrayLikeFloat_co


def deg_to_rad(deg: float):
    return (2 * np.pi * deg) / 360


def rad_to_deg(rad: float):
    return (rad * 360) / (2 * np.pi)


def is_square_matrix(arr: NDArray[Any]):
    shape = arr.shape
    return len(shape) == 2 and shape[0] == shape[1]


def is_symmetric_matrix(arr: NDArray[Any], rtol=1e-05, atol=1e-08):
    return np.allclose(arr, arr.T, rtol, atol)


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

    def to_patch(self, **kwargs) -> EllipsePatch:
        return EllipsePatch(
            (0, 0),
            self.semiaxis_x * 2,
            self.semiaxis_y * 2,
            angle=rad_to_deg(self.angle),
            **kwargs,
        )


@dataclass
class Ellipsoid:
    semiaxis_x: float
    semiaxis_y: float
    semiaxis_z: float
    polar_angle: float
    azimuthal_angle: float


class Conceptor(np.ndarray):
    @staticmethod
    def make_correlation_matrix(matrix: ArrayLikeFloat, axis: int = 0):
        matrix = np.asarray(matrix)
        if axis == 0:
            matrix = matrix.T

        sample_count = matrix.shape[0]
        return (matrix @ matrix.T) / sample_count

    @staticmethod
    def from_state_matrix(matrix: ArrayLikeFloat, aperture: float = 10, axis: int = 0):
        return Conceptor.from_correlation_matrix(
            Conceptor.make_correlation_matrix(matrix, axis), aperture
        )

    @staticmethod
    def from_correlation_matrix(correlation_matrix: NDArray[Any], aperture: float = 10):
        correlation_matrix = np.asarray(correlation_matrix)
        assert is_square_matrix(correlation_matrix)

        aperture_matrix = (aperture ** (-2)) * eye_like(correlation_matrix)

        conceptor_matrix = (
            la.inv(correlation_matrix + aperture_matrix) @ correlation_matrix
        )

        return Conceptor(conceptor_matrix, aperture)

    @staticmethod
    def estimate_aperture(upper: float, tol: float = 1e-2):
        """data/aperture_estimation.jpg"""
        return sqrt(1 - tol) / (sqrt(upper) * sqrt(tol))

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

    @property
    def neg(self) -> Conceptor:
        return eye_like(self) - self  # type: ignore

    @property
    def quota(self) -> float:
        return np.trace(self) / self.order

    def conj(self, other: Conceptor) -> Conceptor:
        return (self.inv + other.inv + eye_like(self)).inv  # type: ignore

    def disj(self, other: Conceptor) -> Conceptor:
        id = eye_like(self)
        return (
            id + (self @ (id - self).inv + other @ (id - other).inv).inv  # type: ignore
        ).inv

    def set_aperture(self, aperture) -> None:
        assert hasattr(self, "aperture")

        ratio = aperture / self.aperture
        self = self @ (self + (ratio**2 * self.neg).inv)

    def _make_ellipsoid(self, principal_axes, rotation, resolution) -> np.ndarray:
        u = np.linspace(0.0, 2.0 * np.pi, resolution)
        v = np.linspace(0.0, np.pi, resolution)

        x = principal_axes[0] * np.outer(np.cos(u), np.sin(v))
        y = principal_axes[1] * np.outer(np.sin(u), np.sin(v))
        z = principal_axes[2] * np.outer(np.ones_like(u), np.cos(v))
        ellipsoid = np.stack((x, y, z), axis=-1)

        if self.order > 3:
            rotation = rotation[:3, :3]

        return (ellipsoid @ rotation).transpose(2, 0, 1)

    def ellipsoid(self, resolution: int = 100) -> np.ndarray:
        _, s, rotation = la.svd(self)
        return self._make_ellipsoid(1.0 / np.sqrt(s), rotation, resolution)

    def ellipsoid_unit(self, resolution: int = 100) -> np.ndarray:
        _, s, rotation = la.svd(self)
        return self._make_ellipsoid(s, rotation, resolution)

    def ellipsoid_sqrt(self, resolution: int = 100) -> np.ndarray:
        _, s, rotation = la.svd(self)
        return self._make_ellipsoid(np.sqrt(s), rotation, resolution)

    def ellipse(self) -> Ellipse:
        """https://en.wikipedia.org/wiki/Ellipsoid#As_a_quadric"""
        eigenvalues, eigenvectors = la.eig(self)
        semiaxes = tuple(eigenvalues)

        ellipse_principal_axis_x = eigenvectors[:2, 0]
        unit_axis_x = np.array([0, 1])

        angle_rad = angle_between_vectors(ellipse_principal_axis_x, unit_axis_x)

        return Ellipse(semiaxes[0], semiaxes[1], angle_rad)


def loewner(a: Conceptor, b: Conceptor, tol: float = 1e-3) -> int:
    """Checks whether the given two conceptors are loewner-ordered.

    Parameters
    ----------
    a : Conceptor
        The first conceptor
    b : Conceptor
        The second conceptor
    tol : float, optional
        The floating-point comparison tolerance, by default 1e-3

    Returns
    -------
    int
        1 iff a >= b; -1 iff a <= b; 0 if no loewner ordering is present between a and b
    """
    diff = a - b
    diff_eigvals = la.eigvals(diff)
    if np.all(diff_eigvals + tol >= 0):
        return 1
    if np.all(diff_eigvals - tol <= 0):
        return -1
    return 0


def posneg_count_fraction(a: Conceptor, b: Conceptor, tol: float = 1e-3) -> float:
    """Fuzzy estimator of the loewner ordering between two conceptors, looking only at the
    amount of positive and negative eigenvalues present in the difference matrix.

    Parameters
    ----------
    a : Conceptor
        The first conceptor
    b : Conceptor
        The second conceptor
    tol : float, optional
        The bound within values are treated as zero, to account for floating-point
        imprecision, by default 1e-3

    Returns
    -------
    float
        The fuzzy loewner-ordering estimation. If below zero, then a <= b up to that
        extent. If above zero, then a >= b up to that extent.
    """
    diff = a - b
    diff_eigvals = np.sort(la.eigvals(diff))

    where_negative = np.argwhere(diff_eigvals < -tol).flatten()
    where_positive = np.argwhere(diff_eigvals > tol).flatten()

    negative_amount = where_negative[-1] if len(where_negative) > 0 else 0
    positives_start_idx = where_positive[0] if len(where_positive) > 0 else 0
    positive_amount = len(diff_eigvals) - positives_start_idx

    fraction = positive_amount / (positive_amount + negative_amount)
    return 2 * fraction - 1


def posneg_magnitude_fraction(a: Conceptor, b: Conceptor, tol: float = 1e-3) -> float:
    """Fuzzy estimator of the loewner ordering between two conceptors, looking at the
    magnitude of all nonzero

    Parameters
    ----------
    a : Conceptor
        The first conceptor
    b : Conceptor
        The second conceptor
    tol : float, optional
        The bound within values are treated as zero, to account for floating-point
        imprecision, by default 1e-3

    Returns
    -------
    float
        The fuzzy loewner-ordering estimation. If below zero, then a <= b up to that
        extent. If above zero, then a >= b up to that extent.
    """
    diff = a - b
    diff_eigvals = np.sort(la.eigvals(diff))

    where_negative = np.argwhere(diff_eigvals < -tol)
    where_positive = np.argwhere(diff_eigvals > tol)

    positive_mean = diff_eigvals[where_positive].sum() if len(where_positive) > 0 else 0
    negative_mean = diff_eigvals[where_negative].sum() if len(where_negative) > 0 else 0

    fraction = positive_mean / (positive_mean - negative_mean)
    return 2 * fraction - 1


def conj(c1: Conceptor, *conceptors: Conceptor):
    ret = c1
    for c in conceptors:
        ret = ret.conj(c)
    return ret


def disj(c1: Conceptor, *conceptors: Conceptor):
    ret = c1
    for c in conceptors:
        ret = ret.disj(c)
    return ret


# %%
