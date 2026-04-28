from collections.abc import Callable

import stim
from sympy.logic.boolalg import Xor, true

from stimsymb.tableau import SymbolicTableau


GateFn = Callable[[SymbolicTableau, int], None]
LocalPauliMap = tuple[tuple[int, int, bool], ...]

SINGLE_QUBIT_CLIFFORD_GATES = (
    "I",
    "X",
    "Y",
    "Z",
    "H",
    "S",
    "S_DAG",
    "SQRT_X",
    "SQRT_X_DAG",
    "SQRT_Y",
    "SQRT_Y_DAG",
    "H_XY",
    "H_YZ",
    "H_NXY",
    "H_NXZ",
    "H_NYZ",
    "C_XYZ",
    "C_ZYX",
    "C_NXYZ",
    "C_NZYX",
    "C_XNYZ",
    "C_XYNZ",
    "C_ZNYX",
    "C_ZYNX",
)


def x(tableau: SymbolicTableau, qubit: int) -> None:
    """Apply an X gate to a symbolic tableau in place."""
    _apply_single_qubit_clifford(tableau, qubit, "X")


def y(tableau: SymbolicTableau, qubit: int) -> None:
    """Apply a Y gate to a symbolic tableau in place."""
    _apply_single_qubit_clifford(tableau, qubit, "Y")


def z(tableau: SymbolicTableau, qubit: int) -> None:
    """Apply a Z gate to a symbolic tableau in place."""
    _apply_single_qubit_clifford(tableau, qubit, "Z")


def _apply_single_qubit_clifford(
    tableau: SymbolicTableau,
    qubit: int,
    gate_name: str,
) -> None:
    _check_qubit(tableau, qubit)
    local_map = _LOCAL_PAULI_MAPS[gate_name]

    for row in range(2 * tableau.num_qubits):
        index = int(tableau.xs[row, qubit] + 2 * tableau.zs[row, qubit])
        new_x, new_z, flips_phase = local_map[index]
        tableau.xs[row, qubit] = new_x
        tableau.zs[row, qubit] = new_z
        if flips_phase:
            tableau.phases[row] = Xor(tableau.phases[row], true)


def execute(tableau: SymbolicTableau, circuit: stim.Circuit) -> None:
    """Execute a supported Stim circuit on a symbolic tableau in place."""
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            for _ in range(instruction.repeat_count):
                execute(tableau, instruction.body_copy())
            continue

        gate = _GATES.get(instruction.name)
        if gate is None:
            raise NotImplementedError(f"unsupported gate: {instruction.name}")
        for target in instruction.targets_copy():
            qubit = target.qubit_value
            if qubit is None:
                raise NotImplementedError("only qubit targets are supported")
            gate(tableau, qubit)


def _check_qubit(tableau: SymbolicTableau, qubit: int) -> None:
    if qubit < 0 or qubit >= tableau.num_qubits:
        raise IndexError("qubit index out of range")


def _local_pauli_map(gate_name: str) -> LocalPauliMap:
    gate = stim.Tableau.from_named_gate(gate_name)
    entries = []
    for pauli in ["_", "X", "Z", "Y"]:
        out = stim.PauliString(pauli).after(gate, [0])
        xs, zs = out.to_numpy()
        entries.append((int(xs[0]), int(zs[0]), out.sign == -1))
    return tuple(entries)


_LOCAL_PAULI_MAPS = {
    gate_name: _local_pauli_map(gate_name)
    for gate_name in SINGLE_QUBIT_CLIFFORD_GATES
}

_GATES: dict[str, GateFn] = {
    "X": x,
    "Y": y,
    "Z": z,
    **{
        gate_name: lambda tableau, qubit, name=gate_name: _apply_single_qubit_clifford(
            tableau,
            qubit,
            name,
        )
        for gate_name in SINGLE_QUBIT_CLIFFORD_GATES
        if gate_name not in {"X", "Y", "Z"}
    },
}


__all__ = ["SINGLE_QUBIT_CLIFFORD_GATES", "execute", "x", "y", "z"]
