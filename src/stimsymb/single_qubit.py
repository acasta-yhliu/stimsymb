from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import cast

import numpy as np
import stim
from numpy.typing import NDArray
from sympy import Symbol
from sympy.logic.boolalg import Boolean, Xor, false, true

from stimsymb.tableau import SymbolicTableau

_LOCAL_PAULIS = ("_", "X", "Z", "Y")
_SINGLE_QUBIT_PAULI_GATES = {"X": (1, 0), "Y": (1, 1), "Z": (0, 1)}

SINGLE_QUBIT_GATES = tuple(
    sorted(
        name
        for name, data in stim.gate_data().items()
        if data.is_unitary and data.is_single_qubit_gate
    )
)

SINGLE_QUBIT_MEASUREMENTS = tuple(
    sorted(
        name
        for name, data in stim.gate_data().items()
        if (
            data.is_single_qubit_gate
            and not data.is_unitary
            and name in {"M", "MR", "MRX", "MRY", "MX", "MY", "R", "RX", "RY"}
        )
    )
)
SINGLE_QUBIT_ERRORS = (
    "DEPOLARIZE1",
    "HERALDED_ERASE",
    "HERALDED_PAULI_CHANNEL_1",
    "I_ERROR",
    "PAULI_CHANNEL_1",
    "X_ERROR",
    "Y_ERROR",
    "Z_ERROR",
)
_SINGLE_QUBIT_MEASUREMENT_BASIS = {
    "M": "Z",
    "MR": "Z",
    "MRX": "X",
    "MRY": "Y",
    "MX": "X",
    "MY": "Y",
    "R": "Z",
    "RX": "X",
    "RY": "Y",
}
_SINGLE_QUBIT_MEASUREMENT_RESET_MEASUREMENTS = {
    "MR": "M",
    "MRX": "MX",
    "MRY": "MY",
    "R": "M",
    "RX": "MX",
    "RY": "MY",
}
_SINGLE_QUBIT_MEASUREMENT_RESET_CORRECTIONS = {
    "MR": "X",
    "MRX": "Z",
    "MRY": "Z",
    "R": "X",
    "RX": "Z",
    "RY": "Z",
}

__all__ = [
    "SINGLE_QUBIT_GATES",
    "SINGLE_QUBIT_ERRORS",
    "SINGLE_QUBIT_MEASUREMENTS",
    "SingleQubitLocalPauliMap",
    "apply_conditional_single_qubit_pauli",
    "apply_single_qubit_error",
    "apply_single_qubit_gate",
    "apply_single_qubit_measurement_maybe_reset",
]


@dataclass(frozen=True, slots=True)
class SingleQubitLocalPauliMap:
    """Compact lookup table for one-qubit Pauli conjugation."""

    entries: NDArray[np.uint8]

    def __post_init__(self) -> None:
        if self.entries.shape != (len(_LOCAL_PAULIS), 3):
            raise ValueError("single-qubit local Pauli map must have shape (4, 3)")
        if self.entries.dtype != np.uint8:
            raise TypeError("single-qubit local Pauli map must have uint8 dtype")

    def __getitem__(self, index: int) -> tuple[int, int, bool]:
        row = self.entries[index]
        return int(row[0]), int(row[1]), bool(row[2])

    @classmethod
    @cache
    def from_named_gate(cls, gate_name: str) -> SingleQubitLocalPauliMap:
        """Return the one-qubit Pauli conjugation map induced by a Stim gate."""
        gate = stim.Tableau.from_named_gate(gate_name)
        entries = np.zeros((len(_LOCAL_PAULIS), 3), dtype=np.uint8)
        for row, pauli in enumerate(_LOCAL_PAULIS):
            # Enumerate I, X, Z, Y in the same order used by the tableau index
            # x + 2z, so a row's local support can index directly into this table.
            out = stim.PauliString(pauli).after(gate, [0])
            # Stim gives the conjugated Pauli support as separate X/Z indicator bits.
            xs, zs = out.to_numpy()
            # Column 0 stores the output X bit.
            entries[row, 0] = xs[0]
            # Column 1 stores the output Z bit.
            entries[row, 1] = zs[0]
            # Column 2 stores whether conjugation introduced a minus sign.
            entries[row, 2] = out.sign == -1
        return cls(entries)

    def apply(self, tableau: SymbolicTableau, qubit: int) -> None:
        """Apply this local Pauli map to one tableau qubit column."""
        # Compute each row's local Pauli label in one vectorized pass.
        indices = tableau.xs[:, qubit] + 2 * tableau.zs[:, qubit]
        transformed = self.entries[indices]

        # Rewrite the target column's X/Z support from the gathered map rows.
        tableau.xs[:, qubit] = transformed[:, 0]
        tableau.zs[:, qubit] = transformed[:, 1]

        # Symbolic phase bits are Python objects, so only this final toggle
        # step remains as a small row loop.
        for row in np.flatnonzero(transformed[:, 2]):
            tableau.phases[int(row)] = Xor(tableau.phases[int(row)], true)


def apply_single_qubit_gate(
    tableau: SymbolicTableau,
    gate_name: str,
    qubit: int,
) -> None:
    """Apply a supported single-qubit Clifford gate to a tableau in place."""
    if qubit < 0 or qubit >= tableau.num_qubits:
        raise IndexError("qubit index out of range")
    if gate_name not in SINGLE_QUBIT_GATES:
        raise NotImplementedError(f"unsupported gate: {gate_name}")

    # The local map is indexed by (x + 2z): I, X, Z, Y.
    SingleQubitLocalPauliMap.from_named_gate(gate_name).apply(tableau, qubit)


def _apply_single_qubit_measurement(
    tableau: SymbolicTableau,
    gate_name: str,
    qubit: int,
    result_symbol: Boolean,
) -> Boolean:
    """Apply a Pauli measurement to a tableau and return its result.

    Nondeterministic measurements introduce ``result_symbol``.
    """
    if qubit < 0 or qubit >= tableau.num_qubits:
        raise IndexError("qubit index out of range")
    if gate_name not in SINGLE_QUBIT_MEASUREMENTS:
        raise NotImplementedError(f"unsupported measurement gate: {gate_name}")

    basis = _SINGLE_QUBIT_MEASUREMENT_BASIS[gate_name]

    # Encode the measured single-qubit Pauli as one binary symplectic row.
    measured_xs = np.zeros(tableau.num_qubits, dtype=np.uint8)
    measured_zs = np.zeros(tableau.num_qubits, dtype=np.uint8)
    measured_xs[qubit] = basis in {"X", "Y"}
    measured_zs[qubit] = basis in {"Y", "Z"}

    # Symplectic product 1 means the tableau row anticommutes with the
    # measured Pauli. A measurement is random exactly when some stabilizer
    # generator anticommutes with the measured Pauli.
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
        # Deterministic case: the measured Pauli is already in the stabilizer
        # group. Its result is the phase of that stabilizer-group element.
        xs = np.zeros(tableau.num_qubits, dtype=np.uint8)
        zs = np.zeros(tableau.num_qubits, dtype=np.uint8)
        phase: Boolean = false

        # Destabilizers tell which stabilizer rows multiply to the measured
        # Pauli: if D_i anticommutes with M, include stabilizer S_i.
        for row in range(tableau.num_qubits):
            if products[row]:
                stabilizer = tableau.num_qubits + row
                # Multiplying Hermitian Pauli rows XORs support and may add a
                # -1 sign depending on Pauli multiplication order.
                zx = int(zs @ tableau.xs[stabilizer])
                xz = int(xs @ tableau.zs[stabilizer])
                xs ^= tableau.xs[stabilizer]
                zs ^= tableau.zs[stabilizer]
                phase = Xor(
                    phase,
                    tableau.phases[stabilizer],
                    true if (zx - xz) % 4 == 2 else false,
                )
        result = phase
    else:
        # Nondeterministic case: introduce a fresh symbolic outcome and update the
        # tableau to make the measured Pauli a new stabilizer generator.
        result = result_symbol

        # The pivot stabilizer anticommutes with the measurement. Its paired
        # destabilizer slot will receive the old pivot row.
        destabilizer = pivot - tableau.num_qubits

        # Clear anticommutation from every non-pivot row by multiplying with the
        # pivot stabilizer. After this, only the pivot row is replaced.
        for row in range(2 * tableau.num_qubits):
            if row not in {pivot, destabilizer} and products[row]:
                tableau.multiply_row(row, pivot)

        # Preserve canonical destabilizer/stabilizer pairing by moving the old
        # pivot stabilizer into its corresponding destabilizer row.
        tableau.xs[destabilizer] = tableau.xs[pivot].copy()
        tableau.zs[destabilizer] = tableau.zs[pivot].copy()
        tableau.phases[destabilizer] = tableau.phases[pivot]

        # The measured Pauli becomes the new stabilizer with symbolic phase/result.
        tableau.xs[pivot] = measured_xs
        tableau.zs[pivot] = measured_zs
        tableau.phases[pivot] = result

    return result


def apply_single_qubit_measurement_maybe_reset(
    tableau: SymbolicTableau,
    gate_name: str,
    qubit: int,
    result_symbol: Boolean,
) -> Boolean:
    """Apply a single-qubit measurement-like gate, including reset variants."""
    if gate_name not in SINGLE_QUBIT_MEASUREMENTS:
        raise NotImplementedError(f"unsupported measurement gate: {gate_name}")

    measurement_gate = _SINGLE_QUBIT_MEASUREMENT_RESET_MEASUREMENTS.get(
        gate_name,
        gate_name,
    )
    result = _apply_single_qubit_measurement(
        tableau,
        measurement_gate,
        qubit,
        result_symbol,
    )
    correction_gate = _SINGLE_QUBIT_MEASUREMENT_RESET_CORRECTIONS.get(gate_name)
    if correction_gate is not None:
        apply_conditional_single_qubit_pauli(
            tableau,
            correction_gate,
            qubit,
            result,
        )
    return result


def apply_single_qubit_error(
    tableau: SymbolicTableau,
    gate_name: str,
    qubit: int,
    condition: Boolean,
    probabilities: tuple[float, float, float, float] | None = None,
) -> dict[Boolean, float] | None:
    """Apply a single-qubit Pauli error conditioned on a symbolic Boolean."""
    if gate_name in {
        "DEPOLARIZE1",
        "HERALDED_ERASE",
        "HERALDED_PAULI_CHANNEL_1",
        "PAULI_CHANNEL_1",
    }:
        if probabilities is None:
            raise ValueError("Pauli channel probabilities must be provided")
        return _apply_single_qubit_pauli_channel(
            tableau,
            qubit,
            condition,
            probabilities,
        )
    if gate_name not in SINGLE_QUBIT_ERRORS:
        raise NotImplementedError(f"unsupported single-qubit error gate: {gate_name}")
    if gate_name == "I_ERROR":
        return None
    apply_conditional_single_qubit_pauli(
        tableau,
        gate_name.removesuffix("_ERROR"),
        qubit,
        condition,
    )
    return None


def _apply_single_qubit_pauli_channel(
    tableau: SymbolicTableau,
    qubit: int,
    condition: Boolean,
    probabilities: tuple[float, float, float, float],
) -> dict[Boolean, float]:
    """Apply a categorical X/Y/Z Pauli channel and return its mechanism distribution."""
    _, *pauli_probabilities = probabilities
    mechanisms: dict[Boolean, float] = {}
    for pauli_gate, probability in zip(("X", "Y", "Z"), pauli_probabilities, strict=True):
        # Mechanism symbols are named by extending the event condition name.
        mechanism = cast(Boolean, Symbol(f"{condition}_{pauli_gate}", boolean=True))
        mechanisms[mechanism] = probability
        apply_conditional_single_qubit_pauli(
            tableau,
            pauli_gate,
            qubit,
            mechanism,
        )
    return mechanisms


def apply_conditional_single_qubit_pauli(
    tableau: SymbolicTableau,
    gate_name: str,
    qubit: int,
    condition: Boolean,
) -> None:
    """Apply a Pauli gate conditioned on a symbolic Boolean."""
    if qubit < 0 or qubit >= tableau.num_qubits:
        raise IndexError("qubit index out of range")
    if gate_name not in _SINGLE_QUBIT_PAULI_GATES:
        raise NotImplementedError(f"unsupported conditional Pauli gate: {gate_name}")
    if condition == false:
        return

    gate_x, gate_z = _SINGLE_QUBIT_PAULI_GATES[gate_name]
    for row in range(2 * tableau.num_qubits):
        anticommutes = (
            tableau.xs[row, qubit] * gate_z + tableau.zs[row, qubit] * gate_x
        ) % 2
        if anticommutes:
            tableau.phases[row] = Xor(tableau.phases[row], condition)
