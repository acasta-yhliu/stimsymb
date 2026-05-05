# Stim Instruction Support

## Supported

### Single-Qubit Clifford Gates

Stim single-qubit unitary gates are discovered from `stim.gate_data()` and applied
through their local Pauli maps.

- C_NXYZ
- C_NZYX
- C_XNYZ
- C_XYNZ
- C_XYZ
- C_ZNYX
- C_ZYNX
- C_ZYX
- H
- H_NXY
- H_NXZ
- H_NYZ
- H_XY
- H_YZ
- I
- S
- SQRT_X
- SQRT_X_DAG
- SQRT_Y
- SQRT_Y_DAG
- S_DAG
- X
- Y
- Z

### Measurements

Supported single-qubit measurement instructions are the hardcoded gate names
listed in `SINGLE_QUBIT_MEASUREMENTS_RESETS`. `MPAD` is also supported as
measurement-record padding, but it does not act on the tableau.

- M
- MPAD
- MX
- MY
- R
- RX
- RY
- MR
- MRX
- MRY
- MXX
- MYY
- MPP
- MZZ

### Two-Qubit Clifford Gates

Stim two-qubit unitary gates are discovered from `stim.gate_data()` and applied
through their local Pauli maps.

- CX
- CXSWAP
- CY
- CZ
- CZSWAP
- II
- ISWAP
- ISWAP_DAG
- SQRT_XX
- SQRT_XX_DAG
- SQRT_YY
- SQRT_YY_DAG
- SQRT_ZZ
- SQRT_ZZ_DAG
- SWAP
- SWAPCX
- XCX
- XCY
- XCZ
- YCX
- YCY
- YCZ

### Structure

- DETECTOR
- OBSERVABLE_INCLUDE
- QUBIT_COORDS
- REPEAT
- SHIFT_COORDS
- TICK

### Noise And Errors

- I_ERROR
- DEPOLARIZE1
- HERALDED_ERASE
- HERALDED_PAULI_CHANNEL_1
- PAULI_CHANNEL_1
- X_ERROR
- Y_ERROR
- Z_ERROR

## Unsupported

### Multi-Pauli Unitaries

- SPP
- SPP_DAG

### Measurements


### Noise And Errors

- DEPOLARIZE2
- E
- ELSE_CORRELATED_ERROR
- II_ERROR
- PAULI_CHANNEL_2
