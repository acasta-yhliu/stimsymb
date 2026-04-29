from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
import stim
from numpy.typing import NDArray
from sympy.logic.boolalg import Xor, true

from stimsymb.tableau import SymbolicTableau

_LOCAL_PAULIS = ("_", "X", "Z", "Y")

DOUBLE_QUBIT_GATES = tuple(
    sorted(
        name
        for name, data in stim.gate_data().items()
        if data.is_unitary and data.is_two_qubit_gate
    )
)

__all__ = [
    "DOUBLE_QUBIT_GATES",
    "DoubleQubitLocalPauliMap",
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

    def apply(self, tableau: SymbolicTableau, first_qubit: int, second_qubit: int) -> None:
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
