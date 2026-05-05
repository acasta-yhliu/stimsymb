from __future__ import annotations

import numpy as np
import stim
from numpy.typing import NDArray
from sympy.logic.boolalg import Boolean, Xor, false, true

from stimsymb.tableau import SymbolicTableau

MULTI_QUBIT_MEASUREMENTS = ("MPP",)

__all__ = [
    "MULTI_QUBIT_MEASUREMENTS",
    "apply_pauli_product_measurement",
]


def parse_multi_pauli_targets(
    targets: list[stim.GateTarget],
) -> list[list[stim.GateTarget]]:
    """Split one flattened `MPP` target list into individual Pauli products."""
    products: list[list[stim.GateTarget]] = []
    current: list[stim.GateTarget] = []
    previous_was_combiner = False

    # Stim stores one `MPP` instruction as a flat stream of Pauli targets with
    # combiners between factors of the same product.
    for target in targets:
        if target.is_combiner:
            previous_was_combiner = True
            continue
        # A new Pauli target without a preceding combiner starts a new product.
        if current and not previous_was_combiner:
            products.append(current)
            current = []
        current.append(target)
        previous_was_combiner = False
    if current:
        products.append(current)
    return products


def apply_pauli_product_measurement(
    tableau: SymbolicTableau,
    measured_xs: NDArray[np.uint8],
    measured_zs: NDArray[np.uint8],
    result_symbol: Boolean,
) -> Boolean:
    """Apply a tableau measurement update for one encoded Pauli product."""
    # Symplectic product 1 means a tableau row anticommutes with the measured
    # Pauli product. Random measurements are exactly the cases with such a
    # stabilizer pivot.
    products = (tableau.xs @ measured_zs + tableau.zs @ measured_xs) % 2
    pivot = next(
        (
            row
            for row in range(tableau.num_qubits, 2 * tableau.num_qubits)
            if products[row]
        ),
        None,
    )
    if pivot is None:
        return _deterministic_measurement_result(tableau, products)

    return _nondeterministic_measurement_result(
        tableau,
        measured_xs,
        measured_zs,
        products,
        pivot,
        result_symbol,
    )


def _measurement_row_from_pauli_targets(
    num_qubits: int,
    targets: list[stim.GateTarget],
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], bool]:
    """Return the tableau row and inversion bit for one Pauli product."""
    measured_xs = np.zeros(num_qubits, dtype=np.uint8)
    measured_zs = np.zeros(num_qubits, dtype=np.uint8)
    is_inverted = False

    # Each Pauli target contributes one local X/Z factor to the measured row.
    for target in targets:
        if target.is_combiner:
            continue
        qubit = target.qubit_value
        assert qubit is not None
        if target.is_x_target:
            measured_xs[qubit] = 1
        elif target.is_y_target:
            measured_xs[qubit] = 1
            measured_zs[qubit] = 1
        else:
            measured_zs[qubit] = 1
        is_inverted ^= target.is_inverted_result_target
    return measured_xs, measured_zs, is_inverted


def _deterministic_measurement_result(
    tableau: SymbolicTableau,
    products: NDArray[np.uint8],
) -> Boolean:
    """Return the phase of a Pauli product already in the stabilizer group."""
    xs = np.zeros(tableau.num_qubits, dtype=np.uint8)
    zs = np.zeros(tableau.num_qubits, dtype=np.uint8)
    phase: Boolean = false

    # Destabilizers identify which stabilizer generators multiply to the
    # measured operator: if D_i anticommutes with M, include S_i.
    for row in range(tableau.num_qubits):
        if not products[row]:
            continue
        stabilizer = tableau.num_qubits + row
        # Multiplying Hermitian Pauli rows XORs support and may add a -1 sign
        # depending on the local Pauli multiplication order.
        zx = int(zs @ tableau.xs[stabilizer])
        xz = int(xs @ tableau.zs[stabilizer])
        xs ^= tableau.xs[stabilizer]
        zs ^= tableau.zs[stabilizer]
        phase = Xor(
            phase,
            tableau.phases[stabilizer],
            true if (zx - xz) % 4 == 2 else false,
        )
    return phase


def _nondeterministic_measurement_result(
    tableau: SymbolicTableau,
    measured_xs: NDArray[np.uint8],
    measured_zs: NDArray[np.uint8],
    products: NDArray[np.uint8],
    pivot: int,
    result_symbol: Boolean,
) -> Boolean:
    """Insert a fresh Pauli-product stabilizer with symbolic phase."""
    result = result_symbol
    destabilizer = pivot - tableau.num_qubits

    # Clear anticommutation from every non-pivot row by multiplying with the
    # pivot stabilizer. After this, the measured Pauli can replace the pivot.
    for row in range(2 * tableau.num_qubits):
        if row not in {pivot, destabilizer} and products[row]:
            tableau.multiply_row(row, pivot)

    # Preserve canonical destabilizer/stabilizer pairing by moving the old
    # pivot stabilizer into its paired destabilizer slot.
    tableau.xs[destabilizer] = tableau.xs[pivot].copy()
    tableau.zs[destabilizer] = tableau.zs[pivot].copy()
    tableau.phases[destabilizer] = tableau.phases[pivot]

    # The measured Pauli product becomes the new stabilizer with the symbolic
    # measurement result as its phase.
    tableau.xs[pivot] = measured_xs
    tableau.zs[pivot] = measured_zs
    tableau.phases[pivot] = result
    return result
