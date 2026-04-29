from __future__ import annotations

from dataclasses import dataclass, field

import stim

from sympy.logic.boolalg import Boolean, Xor, false, true

from stimsymb.gates import (
    MEASUREMENT_GATES,
    SINGLE_QUBIT_CLIFFORD_GATES,
    apply_measurement,
    apply_single_qubit_clifford,
)
from stimsymb.tableau import SymbolicTableau

__all__ = ["RecordExpression", "SymbolicState", "execute"]


@dataclass(slots=True)
class RecordExpression:
    """Boolean record expression with Stim arguments."""

    expression: Boolean
    args: tuple[float, ...]


@dataclass(slots=True)
class SymbolicState:
    """Symbolic execution state containing a tableau and measurement results."""

    tableau: SymbolicTableau
    measurements: list[Boolean] = field(default_factory=list)
    detectors: list[RecordExpression] = field(default_factory=list)
    observables: list[RecordExpression] = field(default_factory=list)


def execute(state: SymbolicState, circuit: stim.Circuit) -> None:
    """Execute a supported Stim circuit on a symbolic state in place."""
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            # Stim repeat blocks are structural; execute the body in order for
            # each repetition so measurement ids keep their global order.
            for _ in range(instruction.repeat_count):
                execute(state, instruction.body_copy())
            continue

        # Coordinate and timing annotations do not affect symbolic state.
        if instruction.name in ("QUBIT_COORDS", "SHIFT_COORDS", "TICK"):
            continue

        # Detector-like instructions consume prior measurement records and
        # preserve their Stim arguments as metadata.
        if instruction.name in {"DETECTOR", "OBSERVABLE_INCLUDE"}:
            record = _record_expression(state, instruction)
            if instruction.name == "DETECTOR":
                state.detectors.append(record)
            else:
                state.observables.append(record)
            continue

        if instruction.name == "MPAD":
            for target in instruction.targets_copy():
                # MPAD appends literal measurement bits without touching qubits.
                qubit = target.qubit_value
                if qubit is None:
                    raise NotImplementedError("MPAD only supports literal bit targets")
                if qubit not in {0, 1}:
                    raise ValueError("MPAD targets must be 0 or 1")
                state.measurements.append(true if qubit else false)
            continue

        if instruction.name in MEASUREMENT_GATES:
            for target in instruction.targets_copy():
                # Measurement ids are the current record length: m0, m1, ...
                qubit = target.qubit_value
                if qubit is None:
                    raise NotImplementedError("measurement only supports qubit targets")
                result = apply_measurement(
                    state.tableau,
                    MEASUREMENT_GATES[instruction.name],
                    qubit,
                    len(state.measurements),
                )
                state.measurements.append(result)
            continue

        if instruction.name in SINGLE_QUBIT_CLIFFORD_GATES:
            for target in instruction.targets_copy():
                qubit = target.qubit_value
                if qubit is None:
                    raise NotImplementedError("single-qubit gates only support qubit targets")
                apply_single_qubit_clifford(state.tableau, instruction.name, qubit)
            continue

        raise NotImplementedError(f"unsupported instruction: {instruction.name}")


def _record_expression(
    state: SymbolicState,
    instruction: stim.CircuitInstruction,
) -> RecordExpression:
    """Return the XOR expression and args from measurement-record targets."""
    terms: list[Boolean] = []
    for target in instruction.targets_copy():
        if not target.is_measurement_record_target:
            raise NotImplementedError(f"{instruction.name} only supports rec targets")
        index = len(state.measurements) + target.value
        if index < 0 or index >= len(state.measurements):
            raise IndexError("measurement record target out of range")
        terms.append(state.measurements[index])
    return RecordExpression(
        expression=Xor(*terms) if terms else false,
        args=tuple(instruction.gate_args_copy()),
    )
