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
    SINGLE_QUBIT_ERRORS,
    SINGLE_QUBIT_MEASUREMENTS,
    apply_single_qubit_error,
    apply_single_qubit_gate,
    apply_single_qubit_measurement_maybe_reset,
)
from stimsymb.tableau import SymbolicTableau

__all__ = [
    "ErrorRecord",
    "MeasurementRecord",
    "RecordExpression",
    "SymbolicState",
    "execute",
]


@dataclass(slots=True)
class RecordExpression:
    """Boolean record expression with Stim arguments."""

    expression: Boolean
    args: tuple[float, ...]


@dataclass(slots=True)
class MeasurementRecord:
    """Visible and latent measurement symbols with their distributions."""

    recorded: list[Boolean] = field(default_factory=list)
    latent: list[Boolean] = field(default_factory=list)
    distribution: dict[Boolean, float] = field(default_factory=dict)

    @property
    def defined_symbols(self) -> list[Boolean]:
        """All measurement-side symbols introduced by execution."""
        return self.recorded + self.latent


@dataclass(slots=True)
class ErrorRecord:
    """Error event symbols with their distributions."""

    events: list[Boolean] = field(default_factory=list)
    distribution: dict[Boolean, float | dict[Boolean, float]] = field(
        default_factory=dict
    )
    mechanism: list[Boolean] = field(default_factory=list)
    all_symbols: list[Boolean] = field(default_factory=list)

    @property
    def defined_symbols(self) -> list[Boolean]:
        """All error-side symbols introduced by execution."""
        return self.all_symbols


@dataclass(slots=True)
class SymbolicState:
    """Symbolic execution state containing a tableau and measurement results."""

    # Current symbolic stabilizer tableau of the quantum state.
    tableau: SymbolicTableau
    # Visible and latent measurement symbols produced by execution.
    measurements: MeasurementRecord = field(default_factory=MeasurementRecord)
    # Error event symbols introduced by noise instructions and report flips.
    errors: ErrorRecord = field(default_factory=ErrorRecord)
    # Recorded detector expressions in circuit order.
    detectors: list[RecordExpression] = field(default_factory=list)
    # Recorded observable-include expressions in circuit order.
    observables: list[RecordExpression] = field(default_factory=list)
    # Monotone executed-instruction counter used for symbol naming.
    _next_instruction_id: int = 0


def execute(state: SymbolicState, circuit: stim.Circuit) -> None:
    """Execute a supported Stim circuit on a symbolic state in place."""
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            # Stim repeat blocks are structural; execute the body in order for
            # each repetition so measurement ids keep their global order.
            for _ in range(instruction.repeat_count):
                execute(state, instruction.body_copy())
            continue

        instruction_id = state._next_instruction_id
        state._next_instruction_id += 1

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
                state.measurements.recorded.append(true if target.qubit_value else false)
            continue

        if instruction.name in SINGLE_QUBIT_MEASUREMENTS:
            gate_args = instruction.gate_args_copy()
            for target_index, target in enumerate(instruction.targets_copy()):
                qubit = target.qubit_value
                assert qubit is not None
                # Plain measurements append to the public record; resets keep
                # their outcomes latent so later noise can reuse the same path.
                is_latent = instruction.name in {"R", "RX", "RY"}
                # Visible and latent outcomes use separate symbol namespaces.
                result_symbol = (
                    _symbol("l", instruction_id, target_index)
                    if is_latent
                    else _symbol("m", len(state.measurements.recorded))
                )
                result = apply_single_qubit_measurement_maybe_reset(
                    state.tableau,
                    instruction.name,
                    qubit,
                    result_symbol,
                )
                # Fresh nondeterministic measurement outcomes inherit the
                # default Bernoulli distribution before any report noise is
                # applied to the visible bit.
                if result == result_symbol and result not in {false, true}:
                    state.measurements.distribution[result] = 0.5
                # Stim's measurement noise flips only the reported bit. For
                # demolition measurements, the reset still uses the true
                # outcome already consumed by the tableau helper.
                if gate_args:
                    result = _apply_measurement_noise(
                        state,
                        instruction_id,
                        target_index,
                        result,
                        gate_args[0],
                    )
                # Resets stash their outcomes internally, while measurements
                # expose them through Stim's measurement record.
                if is_latent:
                    state.measurements.latent.append(result)
                else:
                    state.measurements.recorded.append(result)
            continue

        if instruction.name in SINGLE_QUBIT_ERRORS:
            gate_args = instruction.gate_args_copy()
            error_probability = gate_args[0]
            if not 0 <= error_probability <= 1:
                raise ValueError("single-qubit Pauli error probability must be in [0, 1]")
            pauli_gate = instruction.name.removesuffix("_ERROR")
            for target_index, target in enumerate(instruction.targets_copy()):
                qubit = target.qubit_value
                assert qubit is not None
                error_symbol = _symbol("e", instruction_id, target_index, pauli_gate)
                if error_symbol not in {false, true}:
                    state.errors.distribution[error_symbol] = error_probability
                state.errors.events.append(error_symbol)
                state.errors.mechanism.append(error_symbol)
                state.errors.all_symbols.append(error_symbol)
                apply_single_qubit_error(
                    state.tableau,
                    instruction.name,
                    qubit,
                    error_symbol,
                )
            continue

        if instruction.name in SINGLE_QUBIT_GATES:
            for target in instruction.targets_copy():
                qubit = target.qubit_value
                assert qubit is not None
                apply_single_qubit_gate(state.tableau, instruction.name, qubit)
            continue

        if instruction.name in DOUBLE_QUBIT_GATES:
            targets = instruction.targets_copy()
            for first_target, second_target in zip(
                targets[0::2],
                targets[1::2],
                strict=True,
            ):
                first_qubit = first_target.qubit_value
                second_qubit = second_target.qubit_value
                assert first_qubit is not None
                assert second_qubit is not None
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
        index = len(state.measurements.recorded) + target.value
        if index < 0 or index >= len(state.measurements.recorded):
            raise IndexError("measurement record target out of range")
        terms.append(state.measurements.recorded[index])
    return RecordExpression(
        expression=Xor(*terms) if terms else false,
        args=tuple(instruction.gate_args_copy()),
    )
def _apply_measurement_noise(
    state: SymbolicState,
    instruction_id: int,
    target_index: int,
    result: Boolean,
    error_probability: float,
) -> Boolean:
    """Flip a reported measurement result using a fresh latent error symbol."""
    if not 0 <= error_probability <= 1:
        raise ValueError("measurement noise probability must be in [0, 1]")

    error_symbol = _symbol("e", instruction_id, target_index)
    if error_symbol not in {false, true}:
        state.errors.distribution[error_symbol] = error_probability
    state.errors.events.append(error_symbol)
    state.errors.mechanism.append(error_symbol)
    state.errors.all_symbols.append(error_symbol)
    return Xor(result, error_symbol)


def _symbol(
    prefix: str,
    instruction_id: int,
    target_index: int | None = None,
    postfix: str | None = None,
) -> Boolean:
    """Return a symbolic Boolean with explicit prefix/id/target/postfix parts."""
    parts = [f"{prefix}{instruction_id}"]
    if target_index is not None:
        parts.append(str(target_index))
    if postfix is not None:
        parts.append(postfix)
    return cast(Boolean, Symbol("_".join(parts), boolean=True))
