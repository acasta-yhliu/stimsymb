import random

import numpy as np
import pytest
import stim
from sympy import Symbol
from sympy.logic.boolalg import Boolean, Xor, false, true

from stimsymb.double_qubit import DOUBLE_QUBIT_GATES
from stimsymb.execution import execute, SymbolicState
from stimsymb.single_qubit import (
    SINGLE_QUBIT_GATES,
    SINGLE_QUBIT_MEASUREMENTS,
    apply_single_qubit_measurement_maybe_reset,
)
from stimsymb.tableau import SymbolicTableau


GateSet = tuple[str, ...]


def test_single_qubit_clifford_gates_come_from_stim_metadata() -> None:
    expected = tuple(
        sorted(
            name
            for name, data in stim.gate_data().items()
            if data.is_unitary and data.is_single_qubit_gate
        )
    )

    assert SINGLE_QUBIT_GATES == expected


def test_double_qubit_clifford_gates_come_from_stim_metadata() -> None:
    expected = tuple(
        sorted(
            name
            for name, data in stim.gate_data().items()
            if data.is_unitary and data.is_two_qubit_gate
        )
    )

    assert DOUBLE_QUBIT_GATES == expected


def test_single_qubit_measurements_come_from_stim_metadata() -> None:
    expected = tuple(
        sorted(
            tuple(
                name
                for name, data in stim.gate_data().items()
                if (
                    data.is_single_qubit_gate
                    and not data.is_unitary
                    and name in {"M", "MR", "MRX", "MRY", "MX", "MY", "R", "RX", "RY"}
                )
            )
        )
    )

    assert SINGLE_QUBIT_MEASUREMENTS == expected


def _random_initial_state(num_qubits: int, rng: random.Random) -> stim.Tableau:
    tableau = stim.Tableau(num_qubits)
    for _ in range(4 * num_qubits):
        gate = rng.choice(["H", "S", "CX"])
        if gate == "CX" and num_qubits > 1:
            control, target = rng.sample(range(num_qubits), 2)
            tableau.append(stim.Tableau.from_named_gate(gate), [control, target])
        else:
            tableau.append(
                stim.Tableau.from_named_gate(rng.choice(["H", "S"])),
                [rng.randrange(num_qubits)],
            )
    return tableau


def _random_circuit(
    num_qubits: int,
    num_gates: int,
    gate_set: GateSet,
    rng: random.Random,
) -> stim.Circuit:
    circuit = stim.Circuit()
    for _ in range(num_gates):
        if rng.randrange(4) == 0:
            circuit.append(
                stim.CircuitRepeatBlock(
                    repeat_count=rng.randrange(2, 5),
                    body=_random_flat_circuit(
                        num_qubits=num_qubits,
                        num_gates=rng.randrange(1, 4),
                        gate_set=gate_set,
                        rng=rng,
                    ),
                ),
            )
        else:
            _append_random_gate(circuit, num_qubits, gate_set, rng)
    return circuit


def _random_flat_circuit(
    num_qubits: int,
    num_gates: int,
    gate_set: GateSet,
    rng: random.Random,
) -> stim.Circuit:
    circuit = stim.Circuit()
    for _ in range(num_gates):
        _append_random_gate(circuit, num_qubits, gate_set, rng)
    return circuit


def _append_random_gate(
    circuit: stim.Circuit,
    num_qubits: int,
    gate_set: GateSet,
    rng: random.Random,
) -> None:
    gate = rng.choice(gate_set)
    if gate in DOUBLE_QUBIT_GATES:
        if num_qubits < 2:
            _append_random_gate(circuit, num_qubits, gate_set, rng)
            return
        circuit.append(gate, rng.sample(range(num_qubits), 2), [])
        return

    circuit.append(gate, [rng.randrange(num_qubits)], [])


def _stim_tableau_rows(tableau: stim.Tableau) -> list[stim.PauliString]:
    return [tableau.x_output(q) for q in range(len(tableau))] + [
        tableau.z_output(q) for q in range(len(tableau))
    ]


def _generate_gate_cases() -> list[tuple[int, int, stim.Tableau, stim.Circuit]]:
    cases = []
    for num_qubits in [1, 2, 3, 5]:
        rng = random.Random(num_qubits)
        for case in range(16):
            initial_state = _random_initial_state(num_qubits, rng)
            circuit = _random_circuit(
                num_qubits=num_qubits,
                num_gates=8 * num_qubits,
                gate_set=SINGLE_QUBIT_GATES,
                rng=rng,
            )
            cases.append((num_qubits, case, initial_state, circuit))
    return cases


def _generate_two_qubit_gate_cases() -> list[tuple[int, int, stim.Tableau, stim.Circuit]]:
    cases = []
    gate_set = SINGLE_QUBIT_GATES + DOUBLE_QUBIT_GATES
    for num_qubits in [2, 3, 5]:
        rng = random.Random(100 + num_qubits)
        for case in range(16):
            initial_state = _random_initial_state(num_qubits, rng)
            circuit = _random_circuit(
                num_qubits=num_qubits,
                num_gates=8 * num_qubits,
                gate_set=gate_set,
                rng=rng,
            )
            cases.append((num_qubits, case, initial_state, circuit))
    return cases


@pytest.mark.parametrize(
    ("num_qubits", "initial_state", "circuit"),
    [
        pytest.param(
            num_qubits,
            initial_state,
            circuit,
            id=f"{num_qubits}q-case-{case}",
        )
        for num_qubits, case, initial_state, circuit in _generate_gate_cases()
    ],
)
def test_execute_matches_stim_for_random_initial_state_and_circuit(
    num_qubits: int,
    initial_state: stim.Tableau,
    circuit: stim.Circuit,
) -> None:
    state = SymbolicState(tableau=_from_stim_rows(_stim_tableau_rows(initial_state)))

    execute(state, circuit)
    _assert_output_state_equal(initial_state, circuit, state.tableau)
    assert state.tableau.num_qubits == num_qubits
    assert state.tableau.satisfy_canonical_commutation()


@pytest.mark.parametrize(
    ("num_qubits", "initial_state", "circuit"),
    [
        pytest.param(
            num_qubits,
            initial_state,
            circuit,
            id=f"{num_qubits}q-two-qubit-case-{case}",
        )
        for num_qubits, case, initial_state, circuit in _generate_two_qubit_gate_cases()
    ],
)
def test_execute_two_qubit_clifford_circuits_match_stim(
    num_qubits: int,
    initial_state: stim.Tableau,
    circuit: stim.Circuit,
) -> None:
    state = SymbolicState(tableau=_from_stim_rows(_stim_tableau_rows(initial_state)))

    execute(state, circuit)

    _assert_output_state_equal(initial_state, circuit, state.tableau)
    assert state.tableau.num_qubits == num_qubits
    assert state.tableau.satisfy_canonical_commutation()


@pytest.mark.parametrize(
    ("circuit", "basis"),
    [
        ("M 0", "Z"),
        ("H 0\nMX 0", "X"),
        ("H 0\nS 0\nMY 0", "Y"),
    ],
)
def test_execute_records_deterministic_measurement(circuit: str, basis: str) -> None:
    state = SymbolicState(tableau=SymbolicTableau.zero_state(1))

    execute(state, stim.Circuit(circuit))

    assert state.measurements.recorded == [false]
    assert state.tableau.phases[state.tableau.num_qubits :] == [false]
    assert state.tableau.satisfy_canonical_commutation()


def test_execute_records_symbolic_nondeterministic_measurement() -> None:
    state = SymbolicState(tableau=SymbolicTableau.zero_state(1))

    execute(state, stim.Circuit("H 0\nM 0"))

    assert str(state.measurements.recorded[0]) == "m0"
    assert state.measurements.distribution == {state.measurements.recorded[0]: 0.5}
    assert state.tableau.phases[state.tableau.num_qubits :] == state.measurements.recorded
    assert state.tableau.satisfy_canonical_commutation()


def test_execute_deterministic_measurement_does_not_add_distribution() -> None:
    state = SymbolicState(tableau=SymbolicTableau.zero_state(1))

    execute(state, stim.Circuit("M 0"))

    assert state.measurements.recorded == [false]
    assert state.measurements.distribution == {}


def test_execute_noisy_deterministic_measurement_introduces_latent_error_symbol() -> None:
    state = SymbolicState(tableau=SymbolicTableau.zero_state(1))

    execute(state, stim.Circuit("M(0.125) 0"))

    assert state.measurements.recorded == [Symbol("e0_0", boolean=True)]
    assert state.errors.events == [Symbol("e0_0", boolean=True)]
    assert state.errors.mechanism == [Symbol("e0_0", boolean=True)]
    assert state.errors.all_symbols == [Symbol("e0_0", boolean=True)]
    assert state.errors.distribution == {Symbol("e0_0", boolean=True): 0.125}


def test_execute_noisy_symbolic_measurement_tracks_outcome_and_error_symbols() -> None:
    state = SymbolicState(tableau=SymbolicTableau.zero_state(1))

    execute(state, stim.Circuit("H 0\nM(0.125) 0"))

    assert state.measurements.recorded == [Xor(Symbol("e1_0", boolean=True), Symbol("m0", boolean=True))]
    assert state.errors.events == [Symbol("e1_0", boolean=True)]
    assert state.errors.all_symbols == [Symbol("e1_0", boolean=True)]
    assert state.measurements.distribution == {Symbol("m0", boolean=True): 0.5}
    assert state.errors.distribution == {Symbol("e1_0", boolean=True): 0.125}


def test_execute_zero_probability_measurement_error_still_records_symbol() -> None:
    state = SymbolicState(tableau=SymbolicTableau.zero_state(1))

    execute(state, stim.Circuit("M(0) 0"))

    assert state.measurements.recorded == [Symbol("e0_0", boolean=True)]
    assert state.errors.events == [Symbol("e0_0", boolean=True)]
    assert state.errors.all_symbols == [Symbol("e0_0", boolean=True)]
    assert state.errors.distribution == {Symbol("e0_0", boolean=True): 0.0}


def test_execute_unit_probability_measurement_error_still_records_symbol() -> None:
    state = SymbolicState(tableau=SymbolicTableau.zero_state(1))

    execute(state, stim.Circuit("M(1) 0"))

    assert state.measurements.recorded == [Symbol("e0_0", boolean=True)]
    assert state.errors.events == [Symbol("e0_0", boolean=True)]
    assert state.errors.all_symbols == [Symbol("e0_0", boolean=True)]
    assert state.errors.distribution == {Symbol("e0_0", boolean=True): 1.0}


@pytest.mark.parametrize(
    ("circuit", "reset_instruction_id"),
    [("R 0", 0), ("H 0\nRX 0", 1), ("H 0\nS 0\nRY 0", 2)],
)
def test_execute_records_deterministic_reset_as_latent_result(
    circuit: str,
    reset_instruction_id: int,
) -> None:
    state = SymbolicState(tableau=SymbolicTableau.zero_state(1))

    execute(state, stim.Circuit(circuit))

    assert state.measurements.recorded == []
    assert state.measurements.latent == [false]
    assert state.measurements.distribution == {}
    assert state.tableau.satisfy_canonical_commutation()


@pytest.mark.parametrize(
    ("circuit", "final_measurement", "reset_instruction_id"),
    [
        ("H 0\nR 0", "M", 1),
        ("RX 0", "MX", 0),
        ("RY 0", "MY", 0),
    ],
)
def test_execute_symbolic_reset_uses_latent_symbol_and_prepares_plus_eigenstate(
    circuit: str,
    final_measurement: str,
    reset_instruction_id: int,
) -> None:
    state = SymbolicState(tableau=SymbolicTableau.zero_state(1))

    execute(state, stim.Circuit(f"{circuit}\n{final_measurement} 0"))

    assert state.measurements.latent == [Symbol(f"l{reset_instruction_id}_0", boolean=True)]
    assert state.measurements.recorded == [false]
    assert state.measurements.distribution == {Symbol(f"l{reset_instruction_id}_0", boolean=True): 0.5}
    assert state.tableau.satisfy_canonical_commutation()


@pytest.mark.parametrize(
    ("circuit", "expected_prep"),
    [
        ("MR 0", ""),
        ("H 0\nMRX 0", "H 0"),
        ("H 0\nS 0\nMRY 0", "H 0\nS 0"),
    ],
)
def test_execute_measurement_reset_matches_prepared_plus_eigenstate(
    circuit: str,
    expected_prep: str,
) -> None:
    state = SymbolicState(tableau=SymbolicTableau.zero_state(1))
    expected = SymbolicState(tableau=SymbolicTableau.zero_state(1))

    execute(state, stim.Circuit(circuit))
    execute(expected, stim.Circuit(expected_prep))

    assert len(state.measurements.recorded) == 1
    assert state.measurements.recorded == [false]
    assert state.measurements.distribution == {}
    np.testing.assert_array_equal(state.tableau.xs, expected.tableau.xs)
    np.testing.assert_array_equal(state.tableau.zs, expected.tableau.zs)
    assert state.tableau.phases == expected.tableau.phases
    assert state.tableau.satisfy_canonical_commutation()


def test_execute_noisy_measurement_reset_keeps_reset_fidelity() -> None:
    state = SymbolicState(tableau=SymbolicTableau.zero_state(1))
    expected = SymbolicState(tableau=SymbolicTableau.zero_state(1))

    execute(state, stim.Circuit("MR(0.125) 0"))
    execute(expected, stim.Circuit(""))

    assert state.measurements.recorded == [Symbol("e0_0", boolean=True)]
    assert state.errors.events == [Symbol("e0_0", boolean=True)]
    assert state.errors.all_symbols == [Symbol("e0_0", boolean=True)]
    assert state.errors.distribution == {Symbol("e0_0", boolean=True): 0.125}
    np.testing.assert_array_equal(state.tableau.xs, expected.tableau.xs)
    np.testing.assert_array_equal(state.tableau.zs, expected.tableau.zs)
    assert state.tableau.phases == expected.tableau.phases
    assert state.tableau.satisfy_canonical_commutation()


def test_execute_zero_probability_single_qubit_error_still_records_symbol() -> None:
    state = SymbolicState(tableau=SymbolicTableau.zero_state(1))

    execute(state, stim.Circuit("X_ERROR(0) 0\nM 0"))

    assert state.errors.events == [Symbol("e0_0_X", boolean=True)]
    assert state.errors.all_symbols == [Symbol("e0_0_X", boolean=True)]
    assert state.errors.distribution == {Symbol("e0_0_X", boolean=True): 0.0}
    assert state.measurements.recorded == [Symbol("e0_0_X", boolean=True)]


def test_execute_unit_probability_single_qubit_error_still_records_symbol() -> None:
    state = SymbolicState(tableau=SymbolicTableau.zero_state(1))

    execute(state, stim.Circuit("Y_ERROR(1) 0\nM 0"))

    assert state.errors.events == [Symbol("e0_0_Y", boolean=True)]
    assert state.errors.all_symbols == [Symbol("e0_0_Y", boolean=True)]
    assert state.errors.distribution == {Symbol("e0_0_Y", boolean=True): 1.0}
    assert state.measurements.recorded == [Symbol("e0_0_Y", boolean=True)]


def test_execute_symbolic_single_qubit_error_records_error_symbol_and_flips_measurement() -> None:
    state = SymbolicState(tableau=SymbolicTableau.zero_state(1))

    execute(state, stim.Circuit("X_ERROR(0.125) 0\nM 0"))

    assert state.errors.events == [Symbol("e0_0_X", boolean=True)]
    assert state.errors.mechanism == [Symbol("e0_0_X", boolean=True)]
    assert state.errors.all_symbols == [Symbol("e0_0_X", boolean=True)]
    assert state.errors.distribution == {Symbol("e0_0_X", boolean=True): 0.125}
    assert state.measurements.recorded == [Symbol("e0_0_X", boolean=True)]


def test_execute_multi_target_single_qubit_error_uses_distinct_symbols() -> None:
    state = SymbolicState(tableau=SymbolicTableau.zero_state(2))

    execute(state, stim.Circuit("X_ERROR(0.125) 0 1"))

    assert state.errors.events == [
        Symbol("e0_0_X", boolean=True),
        Symbol("e0_1_X", boolean=True),
    ]
    assert state.errors.all_symbols == [
        Symbol("e0_0_X", boolean=True),
        Symbol("e0_1_X", boolean=True),
    ]
    assert state.errors.distribution == {
        Symbol("e0_0_X", boolean=True): 0.125,
        Symbol("e0_1_X", boolean=True): 0.125,
    }


def test_execute_identity_error_records_event_without_changing_tableau() -> None:
    state = SymbolicState(tableau=SymbolicTableau.zero_state(1))
    xs = state.tableau.xs.copy()
    zs = state.tableau.zs.copy()
    phases = state.tableau.phases.copy()

    execute(state, stim.Circuit("I_ERROR(0.125) 0"))

    assert state.errors.events == [Symbol("e0_0_I", boolean=True)]
    assert state.errors.all_symbols == [Symbol("e0_0_I", boolean=True)]
    assert state.errors.distribution == {Symbol("e0_0_I", boolean=True): 0.125}
    np.testing.assert_array_equal(state.tableau.xs, xs)
    np.testing.assert_array_equal(state.tableau.zs, zs)
    assert state.tableau.phases == phases


@pytest.mark.parametrize(
    ("circuit", "expected_prep"),
    [
        ("H 0\nMR 0", ""),
        ("MRX 0", "H 0"),
        ("MRY 0", "H 0\nS 0"),
    ],
)
def test_execute_symbolic_measurement_reset_records_result_and_resets_state(
    circuit: str,
    expected_prep: str,
) -> None:
    state = SymbolicState(tableau=SymbolicTableau.zero_state(1))
    expected = SymbolicState(tableau=SymbolicTableau.zero_state(1))

    execute(state, stim.Circuit(circuit))
    execute(expected, stim.Circuit(expected_prep))

    assert len(state.measurements.recorded) == 1
    assert str(state.measurements.recorded[0]) == "m0"
    assert state.measurements.distribution == {state.measurements.recorded[0]: 0.5}
    np.testing.assert_array_equal(state.tableau.xs, expected.tableau.xs)
    np.testing.assert_array_equal(state.tableau.zs, expected.tableau.zs)
    assert state.tableau.phases == expected.tableau.phases
    assert state.tableau.satisfy_canonical_commutation()


def test_execute_records_mpad_without_changing_tableau() -> None:
    state = SymbolicState(tableau=SymbolicTableau.zero_state(1))
    xs = state.tableau.xs.copy()
    zs = state.tableau.zs.copy()
    phases = state.tableau.phases.copy()

    execute(state, stim.Circuit("MPAD 0 1 0"))

    assert state.measurements.recorded == [false, true, false]
    np.testing.assert_array_equal(state.tableau.xs, xs)
    np.testing.assert_array_equal(state.tableau.zs, zs)
    assert state.tableau.phases == phases


def test_execute_ignores_metadata_instructions() -> None:
    state = SymbolicState(tableau=SymbolicTableau.zero_state(2))
    xs = state.tableau.xs.copy()
    zs = state.tableau.zs.copy()
    phases = state.tableau.phases.copy()

    execute(
        state,
        stim.Circuit(
            """
            QUBIT_COORDS(1, 2) 0
            SHIFT_COORDS(3, 4)
            TICK
            """
        ),
    )

    assert state.measurements.recorded == []
    np.testing.assert_array_equal(state.tableau.xs, xs)
    np.testing.assert_array_equal(state.tableau.zs, zs)
    assert state.tableau.phases == phases


def test_execute_records_detector_expressions() -> None:
    state = SymbolicState(tableau=SymbolicTableau.zero_state(2))

    execute(
        state,
        stim.Circuit(
            """
            M 0
            H 1
            M 1
            DETECTOR(1.5, 2, 3) rec[-1] rec[-2]
            """
        ),
    )

    assert state.measurements.recorded[0] == false
    assert str(state.measurements.recorded[1]) == "m1"
    assert len(state.detectors) == 1
    assert state.detectors[0].expression == state.measurements.recorded[1]
    assert state.detectors[0].args == (1.5, 2.0, 3.0)


def test_execute_records_observable_expressions() -> None:
    state = SymbolicState(tableau=SymbolicTableau.zero_state(2))

    execute(
        state,
        stim.Circuit(
            """
            H 0
            M 0
            MPAD 1
            OBSERVABLE_INCLUDE(3) rec[-2]
            OBSERVABLE_INCLUDE(3) rec[-1]
            """
        ),
    )

    assert str(state.measurements.recorded[0]) == "m0"
    assert len(state.observables) == 2
    assert state.observables[0].expression == state.measurements.recorded[0]
    assert state.observables[0].args == (3.0,)
    assert state.observables[1].expression == true
    assert state.observables[1].args == (3.0,)


@pytest.mark.parametrize(
    ("circuit", "gate_name"),
    [
        ("", "M"),
        ("H 0", "MX"),
        ("H 0\nS 0", "MY"),
    ],
)
def test_measure_returns_deterministic_result_without_symbol(
    circuit: str,
    gate_name: str,
) -> None:
    state = SymbolicState(tableau=SymbolicTableau.zero_state(1))
    execute(state, stim.Circuit(circuit))

    result = apply_single_qubit_measurement_maybe_reset(
        state.tableau,
        gate_name,
        0,
        result_symbol=false,
    )

    assert result == false
    assert state.measurements.recorded == []


@pytest.mark.parametrize("gate_name", ["MX", "MY", "M"])
def test_measure_introduces_symbol_for_nondeterministic_result(gate_name: str) -> None:
    state = SymbolicState(tableau=SymbolicTableau.zero_state(1))

    if gate_name == "M":
        execute(state, stim.Circuit("H 0"))

    result = apply_single_qubit_measurement_maybe_reset(
        state.tableau,
        gate_name,
        0,
        result_symbol=Symbol("m0", boolean=True),
    )

    assert str(result) == "m0"




def _from_stim_rows(rows: list[stim.PauliString]) -> SymbolicTableau:
    num_qubits = len(rows) // 2
    xs = np.zeros((2 * num_qubits, num_qubits), dtype=np.uint8)
    zs = np.zeros((2 * num_qubits, num_qubits), dtype=np.uint8)
    phases: list[Boolean] = []

    for row_index, row in enumerate(rows):
        row_xs, row_zs = row.to_numpy()
        xs[row_index] = row_xs.astype(np.uint8)
        zs[row_index] = row_zs.astype(np.uint8)
        phases.append(true if row.sign == -1 else false)

    return SymbolicTableau(num_qubits=num_qubits, xs=xs, zs=zs, phases=phases)


def _assert_output_state_equal(
    initial_state: stim.Tableau,
    circuit: stim.Circuit,
    tableau: SymbolicTableau,
) -> None:
    rows = [row.after(circuit) for row in _stim_tableau_rows(initial_state)]
    expected_xs = np.zeros_like(tableau.xs)
    expected_zs = np.zeros_like(tableau.zs)
    expected_phases: list[Boolean] = []

    for row_index, row in enumerate(rows):
        row_xs, row_zs = row.to_numpy()
        expected_xs[row_index] = row_xs.astype(np.uint8)
        expected_zs[row_index] = row_zs.astype(np.uint8)
        expected_phases.append(true if row.sign == -1 else false)

    np.testing.assert_array_equal(tableau.xs, expected_xs)
    np.testing.assert_array_equal(tableau.zs, expected_zs)
    assert tableau.phases == expected_phases
