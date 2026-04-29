from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import stim

from sympy import Symbol
from sympy.logic.boolalg import Boolean, Xor, false, true

from stimsymb.double_qubit import (
    DOUBLE_QUBIT_GATES,
    apply_double_qubit_gate,
)
from stimsymb.single_qubit import (
    SINGLE_QUBIT_GATES,
    SINGLE_QUBIT_MEASUREMENTS,
    apply_single_qubit_gate,
    apply_single_qubit_measurement_maybe_reset,
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
    latent_symbols: list[Boolean] = field(default_factory=list)
    distribution: dict[Boolean, float] = field(default_factory=dict)
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

        if instruction.name in SINGLE_QUBIT_MEASUREMENTS:
            for target in instruction.targets_copy():
                qubit = target.qubit_value
                if qubit is None:
                    raise NotImplementedError("measurement only supports qubit targets")
                # Plain measurements append to the public record; resets keep
                # their outcomes latent so later noise can reuse the same path.
                is_latent = instruction.name in {"R", "RX", "RY"}
                # Visible and latent outcomes use separate symbol namespaces.
                result_symbol = (
                    _latent_symbol(len(state.latent_symbols))
                    if is_latent
                    else _measurement_symbol(len(state.measurements))
                )
                result = apply_single_qubit_measurement_maybe_reset(
                    state.tableau,
                    instruction.name,
                    qubit,
                    result_symbol,
                )
                # Only fresh nondeterministic outcomes inherit the default
                # Bernoulli distribution; deterministic results return an
                # existing expression or literal instead.
                if result == result_symbol:
                    _record_distribution(state, result, p_true=0.5)
                # Resets stash their outcomes internally, while measurements
                # expose them through Stim's measurement record.
                if is_latent:
                    state.latent_symbols.append(result)
                else:
                    state.measurements.append(result)
            continue

        if instruction.name in SINGLE_QUBIT_GATES:
            for target in instruction.targets_copy():
                qubit = target.qubit_value
                if qubit is None:
                    raise NotImplementedError(
                        "single-qubit gates only support qubit targets"
                    )
                apply_single_qubit_gate(state.tableau, instruction.name, qubit)
            continue

        if instruction.name in DOUBLE_QUBIT_GATES:
            targets = instruction.targets_copy()
            if len(targets) % 2 != 0:
                raise NotImplementedError(
                    "two-qubit gates require pairs of qubit targets"
                )
            for first_target, second_target in zip(
                targets[0::2],
                targets[1::2],
                strict=True,
            ):
                first_qubit = first_target.qubit_value
                second_qubit = second_target.qubit_value
                if first_qubit is None or second_qubit is None:
                    raise NotImplementedError(
                        "two-qubit gates only support qubit targets"
                    )
                apply_double_qubit_gate(
                    state.tableau,
                    instruction.name,
                    first_qubit,
                    second_qubit,
                )
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


def _record_distribution(state: SymbolicState, symbol: Boolean, p_true: float) -> None:
    """Record the Bernoulli distribution for a newly introduced symbol."""
    if symbol not in {false, true}:
        state.distribution[symbol] = p_true


def _measurement_symbol(measurement_id: int) -> Boolean:
    """Return the fresh symbolic name for a visible measurement outcome."""
    return cast(Boolean, Symbol(f"m{measurement_id}", boolean=True))


def _latent_symbol(latent_id: int) -> Boolean:
    """Return the fresh symbolic name for a latent internal outcome."""
    return cast(Boolean, Symbol(f"l{latent_id}", boolean=True))
