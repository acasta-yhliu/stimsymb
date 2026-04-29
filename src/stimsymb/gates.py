from __future__ import annotations

from typing import cast

import numpy as np
import stim
from sympy import Symbol
from sympy.logic.boolalg import Boolean, Xor, false, true

from stimsymb.tableau import SymbolicTableau

LocalPauliMap = tuple[tuple[int, int, bool], ...]
MeasurementBasis = str
_LOCAL_PAULIS = ("_", "X", "Z", "Y")
MEASUREMENT_GATES = {"M": "Z", "MX": "X", "MY": "Y", "MZ": "Z"}

SINGLE_QUBIT_CLIFFORD_GATES = tuple(
    sorted(
        name
        for name, data in stim.gate_data().items()
        if data.is_unitary and data.is_single_qubit_gate
    )
)

__all__ = [
    "MEASUREMENT_GATES",
    "SINGLE_QUBIT_CLIFFORD_GATES",
    "apply_measurement",
    "apply_single_qubit_clifford",
]


def apply_single_qubit_clifford(
    tableau: SymbolicTableau,
    gate_name: str,
    qubit: int,
) -> None:
    """Apply a supported single-qubit Clifford gate to a tableau in place."""
    if qubit < 0 or qubit >= tableau.num_qubits:
        raise IndexError("qubit index out of range")
    if gate_name not in SINGLE_QUBIT_CLIFFORD_GATES:
        raise NotImplementedError(f"unsupported gate: {gate_name}")

    # The local map is indexed by (x + 2z): I, X, Z, Y.
    local_map = _local_pauli_map(gate_name)
    for row in range(2 * tableau.num_qubits):
        index = int(tableau.xs[row, qubit] + 2 * tableau.zs[row, qubit])
        new_x, new_z, flips_phase = local_map[index]
        tableau.xs[row, qubit] = new_x
        tableau.zs[row, qubit] = new_z
        if flips_phase:
            tableau.phases[row] = Xor(tableau.phases[row], true)


def apply_measurement(
    tableau: SymbolicTableau,
    basis: MeasurementBasis,
    qubit: int,
    measurement_id: int,
) -> Boolean:
    """Apply a Pauli measurement to a tableau and return its result.

    Nondeterministic measurements introduce ``m{measurement_id}``.
    """
    if qubit < 0 or qubit >= tableau.num_qubits:
        raise IndexError("qubit index out of range")
    if basis not in {"X", "Y", "Z"}:
        raise NotImplementedError(f"unsupported measurement basis: {basis}")

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
        return phase

    # Nondeterministic case: introduce a fresh symbolic outcome and update the
    # tableau to make the measured Pauli a new stabilizer generator.
    result = cast(Boolean, Symbol(f"m{measurement_id}", boolean=True))

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


def _local_pauli_map(gate_name: str) -> LocalPauliMap:
    """Return the one-qubit Pauli conjugation map induced by a Stim gate."""
    gate = stim.Tableau.from_named_gate(gate_name)
    entries = []
    for pauli in _LOCAL_PAULIS:
        out = stim.PauliString(pauli).after(gate, [0])
        xs, zs = out.to_numpy()
        entries.append((int(xs[0]), int(zs[0]), out.sign == -1))
    return tuple(entries)
