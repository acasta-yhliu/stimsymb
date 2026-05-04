from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
import stim
from numpy.typing import NDArray
from sympy.logic.boolalg import Boolean, Xor, false, true

from stimsymb.tableau import SymbolicTableau

_LOCAL_PAULIS = ("_", "X", "Z", "Y")
DOUBLE_QUBIT_MEASUREMENTS = ("MXX", "MYY", "MZZ")
_DOUBLE_QUBIT_MEASUREMENT_PAULIS = {
    "MXX": ((1, 0), (1, 0)),
    "MYY": ((1, 1), (1, 1)),
    "MZZ": ((0, 1), (0, 1)),
}

DOUBLE_QUBIT_GATES = tuple(
    sorted(
        name
        for name, data in stim.gate_data().items()
        if data.is_unitary and data.is_two_qubit_gate
    )
)

__all__ = [
    "DOUBLE_QUBIT_GATES",
    "DOUBLE_QUBIT_MEASUREMENTS",
    "DoubleQubitLocalPauliMap",
    "apply_double_qubit_measurement",
    "apply_double_qubit_gate",
]


@dataclass(frozen=True, slots=True)
class DoubleQubitLocalPauliMap:
    """Compact lookup table for two-qubit Pauli conjugation."""

    entries: NDArray[np.uint8]

    def __post_init__(self) -> None:
        if self.entries.shape != (len(_LOCAL_PAULIS) ** 2, 5):
            raise ValueError("two-qubit local Pauli map must have shape (16, 5)")
        if self.entries.dtype != np.uint8:
            raise TypeError("two-qubit local Pauli map must have uint8 dtype")

    def __getitem__(self, index: int) -> tuple[int, int, int, int, bool]:
        row = self.entries[index]
        return int(row[0]), int(row[1]), int(row[2]), int(row[3]), bool(row[4])

    @classmethod
    @cache
    def from_named_gate(cls, gate_name: str) -> DoubleQubitLocalPauliMap:
        """Return the two-qubit Pauli conjugation map induced by a Stim gate."""
        gate = stim.Tableau.from_named_gate(gate_name)
        entries = np.zeros((len(_LOCAL_PAULIS) ** 2, 5), dtype=np.uint8)
        for row, first_pauli in enumerate(_LOCAL_PAULIS):
            for col, second_pauli in enumerate(_LOCAL_PAULIS):
                # Enumerate local Paulis in the same order as the tableau index
                # 4 * (x0 + 2z0) + (x1 + 2z1), so each row can gather directly.
                index = 4 * row + col
                out = stim.PauliString(first_pauli + second_pauli).after(gate, [0, 1])
                # Stim returns the conjugated Pauli support as X/Z indicator bits.
                xs, zs = out.to_numpy()
                # Columns 0-3 store the output X/Z bits for qubits 0 and 1.
                entries[index, 0] = xs[0]
                entries[index, 1] = zs[0]
                entries[index, 2] = xs[1]
                entries[index, 3] = zs[1]
                # Column 4 stores whether conjugation introduced a minus sign.
                entries[index, 4] = out.sign == -1
        return cls(entries)

    def apply(
        self, tableau: SymbolicTableau, first_qubit: int, second_qubit: int
    ) -> None:
        """Apply this local Pauli map to two tableau qubit columns."""
        # Compute each row's two-qubit Pauli label in one vectorized pass.
        indices = (
            4 * (tableau.xs[:, first_qubit] + 2 * tableau.zs[:, first_qubit])
            + tableau.xs[:, second_qubit]
            + 2 * tableau.zs[:, second_qubit]
        )
        transformed = self.entries[indices]

        # Rewrite both target columns' X/Z support from the gathered map rows.
        tableau.xs[:, first_qubit] = transformed[:, 0]
        tableau.zs[:, first_qubit] = transformed[:, 1]
        tableau.xs[:, second_qubit] = transformed[:, 2]
        tableau.zs[:, second_qubit] = transformed[:, 3]

        # Symbolic phase bits are Python objects, so phase toggles stay as a
        # small row loop after the vectorized support update.
        for row in np.flatnonzero(transformed[:, 4]):
            tableau.phases[int(row)] = Xor(tableau.phases[int(row)], true)


def apply_double_qubit_gate(
    tableau: SymbolicTableau,
    gate_name: str,
    first_qubit: int,
    second_qubit: int,
) -> None:
    """Apply a supported two-qubit Clifford gate to a tableau in place."""
    if first_qubit < 0 or first_qubit >= tableau.num_qubits:
        raise IndexError("first qubit index out of range")
    if second_qubit < 0 or second_qubit >= tableau.num_qubits:
        raise IndexError("second qubit index out of range")
    if first_qubit == second_qubit:
        raise ValueError("two-qubit gates require distinct qubits")
    if gate_name not in DOUBLE_QUBIT_GATES:
        raise NotImplementedError(f"unsupported gate: {gate_name}")

    # The local map is indexed by 4 * (x0 + 2z0) + (x1 + 2z1).
    DoubleQubitLocalPauliMap.from_named_gate(gate_name).apply(
        tableau,
        first_qubit,
        second_qubit,
    )


def apply_double_qubit_measurement(
    tableau: SymbolicTableau,
    gate_name: str,
    first_qubit: int,
    second_qubit: int,
    result_symbol: Boolean,
) -> Boolean:
    """Apply a supported two-qubit Pauli-product measurement in place."""
    if first_qubit < 0 or first_qubit >= tableau.num_qubits:
        raise IndexError("first qubit index out of range")
    if second_qubit < 0 or second_qubit >= tableau.num_qubits:
        raise IndexError("second qubit index out of range")
    if first_qubit == second_qubit:
        raise ValueError("two-qubit measurements require distinct qubits")
    if gate_name not in DOUBLE_QUBIT_MEASUREMENTS:
        raise NotImplementedError(f"unsupported measurement gate: {gate_name}")

    measured_xs = np.zeros(tableau.num_qubits, dtype=np.uint8)
    measured_zs = np.zeros(tableau.num_qubits, dtype=np.uint8)
    (first_x, first_z), (second_x, second_z) = _DOUBLE_QUBIT_MEASUREMENT_PAULIS[
        gate_name
    ]
    measured_xs[first_qubit] = first_x
    measured_zs[first_qubit] = first_z
    measured_xs[second_qubit] = second_x
    measured_zs[second_qubit] = second_z
    return _apply_pauli_product_measurement(
        tableau,
        measured_xs,
        measured_zs,
        result_symbol,
    )


def _apply_pauli_product_measurement(
    tableau: SymbolicTableau,
    measured_xs: NDArray[np.uint8],
    measured_zs: NDArray[np.uint8],
    result_symbol: Boolean,
) -> Boolean:
    """Apply a Pauli-product measurement update to a tableau."""
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
