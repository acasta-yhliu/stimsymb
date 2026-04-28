import numpy as np
import pytest
from sympy import Symbol
from sympy.logic.boolalg import false, true

from stimsymb import SymbolicTableau


def test_zero_state_shapes() -> None:
    tableau = SymbolicTableau.zero_state(3)

    assert tableau.num_qubits == 3
    assert tableau.xs.shape == (6, 3)
    assert tableau.zs.shape == (6, 3)
    assert len(tableau.phases) == 6
    assert tableau.destabilizer_xs.shape == (3, 3)
    assert tableau.destabilizer_zs.shape == (3, 3)
    assert tableau.stabilizer_xs.shape == (3, 3)
    assert tableau.stabilizer_zs.shape == (3, 3)
    assert len(tableau.destabilizer_phases) == 3
    assert len(tableau.stabilizer_phases) == 3


def test_zero_state_destabilizers_and_stabilizers() -> None:
    tableau = SymbolicTableau.zero_state(3)

    np.testing.assert_array_equal(tableau.xs[:3], np.eye(3, dtype=np.uint8))
    np.testing.assert_array_equal(tableau.zs[:3], np.zeros((3, 3), dtype=np.uint8))
    np.testing.assert_array_equal(tableau.xs[3:], np.zeros((3, 3), dtype=np.uint8))
    np.testing.assert_array_equal(tableau.zs[3:], np.eye(3, dtype=np.uint8))
    assert tableau.phases == [false] * 6

    np.testing.assert_array_equal(tableau.destabilizer_xs, np.eye(3, dtype=np.uint8))
    np.testing.assert_array_equal(tableau.destabilizer_zs, np.zeros((3, 3), dtype=np.uint8))
    np.testing.assert_array_equal(tableau.stabilizer_xs, np.zeros((3, 3), dtype=np.uint8))
    np.testing.assert_array_equal(tableau.stabilizer_zs, np.eye(3, dtype=np.uint8))
    assert tableau.destabilizer_phases == [false] * 3
    assert tableau.stabilizer_phases == [false] * 3


def test_zero_state_satisfies_canonical_commutation() -> None:
    tableau = SymbolicTableau.zero_state(3)

    assert tableau.satisfy_canonical_commutation()


def test_from_stabilizers_matches_zero_state_support() -> None:
    tableau = SymbolicTableau.from_stabilizers(
        xs=np.zeros((2, 2), dtype=np.uint8),
        zs=np.eye(2, dtype=np.uint8),
    )

    np.testing.assert_array_equal(tableau.xs, SymbolicTableau.zero_state(2).xs)
    np.testing.assert_array_equal(tableau.zs, SymbolicTableau.zero_state(2).zs)
    assert tableau.satisfy_canonical_commutation()


def test_from_stabilizers_builds_dual_destabilizers() -> None:
    tableau = SymbolicTableau.from_stabilizers(
        xs=np.eye(2, dtype=np.uint8),
        zs=np.zeros((2, 2), dtype=np.uint8),
    )

    np.testing.assert_array_equal(
        tableau.destabilizer_xs,
        np.zeros((2, 2), dtype=np.uint8),
    )
    np.testing.assert_array_equal(tableau.destabilizer_zs, np.eye(2, dtype=np.uint8))
    assert tableau.satisfy_canonical_commutation()


def test_from_stabilizers_preserves_symbolic_phases() -> None:
    phase = Symbol("m0", boolean=True)
    tableau = SymbolicTableau.from_stabilizers(
        xs=np.zeros((1, 1), dtype=np.uint8),
        zs=np.eye(1, dtype=np.uint8),
        phases=[phase],
    )

    assert tableau.destabilizer_phases == [false]
    assert tableau.stabilizer_phases == [phase]


def test_multiply_row_updates_support() -> None:
    tableau = SymbolicTableau.zero_state(2)

    tableau.multiply_row(target=2, source=3)

    np.testing.assert_array_equal(tableau.xs[2], np.array([0, 0], dtype=np.uint8))
    np.testing.assert_array_equal(tableau.zs[2], np.array([1, 1], dtype=np.uint8))
    assert tableau.phases[2] == false


def test_multiply_row_propagates_symbolic_phase() -> None:
    phase = Symbol("m0", boolean=True)
    tableau = SymbolicTableau.zero_state(2)
    tableau.phases[3] = phase

    tableau.multiply_row(target=2, source=3)

    assert tableau.phases[2] == phase


def test_multiply_row_tracks_pauli_product_sign() -> None:
    tableau = SymbolicTableau(
        num_qubits=2,
        xs=np.array([[1, 1], [0, 0], [0, 0], [0, 0]], dtype=np.uint8),
        zs=np.array([[0, 0], [1, 1], [0, 0], [0, 0]], dtype=np.uint8),
        phases=[false] * 4,
    )

    tableau.multiply_row(target=0, source=1)

    np.testing.assert_array_equal(tableau.xs[0], np.array([1, 1], dtype=np.uint8))
    np.testing.assert_array_equal(tableau.zs[0], np.array([1, 1], dtype=np.uint8))
    assert tableau.phases[0] == true


def test_multiply_row_rejects_anticommuting_rows() -> None:
    tableau = SymbolicTableau.zero_state(1)

    with pytest.raises(ValueError, match="anticommuting"):
        tableau.multiply_row(target=0, source=1)


def test_detects_broken_canonical_commutation() -> None:
    tableau = SymbolicTableau.zero_state(2)
    tableau.stabilizer_zs[0, 0] = 0

    assert not tableau.satisfy_canonical_commutation()


def test_tableau_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        SymbolicTableau(
            num_qubits=2,
            xs=np.zeros((3, 2), dtype=np.uint8),
            zs=np.zeros((4, 2), dtype=np.uint8),
            phases=[false] * 4,
        )


def test_from_stabilizers_rejects_noncommuting_generators() -> None:
    with pytest.raises(ValueError, match="commute"):
        SymbolicTableau.from_stabilizers(
            xs=np.array([[1, 0], [0, 0]], dtype=np.uint8),
            zs=np.array([[0, 0], [1, 0]], dtype=np.uint8),
        )


def test_from_stabilizers_rejects_dependent_generators() -> None:
    with pytest.raises(ValueError, match="independent"):
        SymbolicTableau.from_stabilizers(
            xs=np.zeros((2, 2), dtype=np.uint8),
            zs=np.array([[1, 0], [1, 0]], dtype=np.uint8),
        )


def test_tableau_rejects_non_binary_values() -> None:
    with pytest.raises(ValueError, match="0 or 1"):
        SymbolicTableau(
            num_qubits=1,
            xs=np.array([[2], [0]], dtype=np.uint8),
            zs=np.zeros((2, 1), dtype=np.uint8),
            phases=[false] * 2,
        )
