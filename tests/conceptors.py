import unittest

import numpy as np

from app.wsd.conceptors.conceptor import (
    Conceptor,
    conj,
    disj,
    is_square_matrix,
    is_symmetric_matrix,
    loewner,
)


class TestConceptors(unittest.TestCase):
    def make_conceptor(
        self, order: int, aperture: int = 1, no_entries: int = 10
    ) -> Conceptor:
        x = np.random.rand(no_entries, order)
        return Conceptor.from_state_matrix(x, aperture)

    def test_conceptor_adheres_to_properties(self):
        states = np.random.rand(23, 5)
        conceptor = Conceptor.from_state_matrix(states, aperture=1)

        self.assertTrue(
            is_square_matrix(conceptor), "Expected conceptor to be a square matrix"
        )
        self.assertTrue(
            is_symmetric_matrix(conceptor), "Expected conceptor to be symmetric"
        )
        self.assertEqual(
            conceptor.shape[0],
            conceptor.order,
            "Expected conceptor order to match its shape",
        )
        self.assertEqual(
            5,
            conceptor.order,
            "Expected conceptor order to match dimensionality of state matrix",
        )
        self.assertTrue(
            np.all(np.linalg.eigvals(conceptor) >= 0),
            "Expected conceptor to have nonnegative eigenvalues",
        )
        self.assertEqual(
            1,
            conceptor.aperture,
            "Expected conceptor aperture to be equal to the constructor parameter",
        )

    def test_disjunction_commutative(self):
        a = self.make_conceptor(10)
        b = self.make_conceptor(10)

        self.assertTrue(
            np.allclose(disj(a, b), disj(b, a)),
            "Expected the disjunction operator to be commutative",
        )

    def test_conjunction_commutative(self):
        a = self.make_conceptor(10)
        b = self.make_conceptor(10)

        self.assertTrue(
            np.allclose(conj(a, b), conj(b, a)),
            "Expected the conjunction operator to be commutative",
        )

    def test_conjunction_less_abstract(self):
        a = self.make_conceptor(10)
        b = self.make_conceptor(10)

        self.assertEqual(
            0, loewner(a, b), "Expected two random matrices to not be loewner-ordered"
        )
        self.assertEqual(
            -1,
            loewner(conj(a, b), a),
            "Expected matrix conjunction to be less abstract than operands",
        )
        self.assertEqual(
            -1,
            loewner(conj(a, b), b),
            "Expected matrix conjunction to be less abstract than operands",
        )

    def test_disjunction_more_abstract(self):
        a = self.make_conceptor(10)
        b = self.make_conceptor(10)

        self.assertEqual(
            0, loewner(a, b), "Expected two random matrices to not be loewner-ordered"
        )
        self.assertEqual(
            1,
            loewner(disj(a, b), a),
            "Expected matrix conjunction to be more abstract than operands",
        )
        self.assertEqual(
            1,
            loewner(disj(a, b), b),
            "Expected matrix conjunction to be more abstract than operands",
        )

    def test_loewner_antisymmetric(self):
        a = self.make_conceptor(10)
        b = a.conj(self.make_conceptor(10))

        self.assertTrue(
            np.allclose(loewner(a, b), loewner(b, a) * -1),
            "Expected the loewner-ordering operator to be antisymmetric",
        )

    def test_aperture_adaptation(self):
        a = self.make_conceptor(10, aperture=1.2)

        b = a.copy()
        b.set_aperture(1000)
        b.set_aperture(1.2)

        self.assertTrue(
            np.allclose(a, b), "Expected aperture adaptation to be reversible"
        )
