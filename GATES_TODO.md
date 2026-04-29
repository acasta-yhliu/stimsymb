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

- M
- MPAD
- MX
- MY
- MZ

### Structure

- DETECTOR
- OBSERVABLE_INCLUDE
- QUBIT_COORDS
- REPEAT
- SHIFT_COORDS
- TICK

## Unsupported

### Two-Qubit Clifford Gates

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

### Multi-Pauli Unitaries

- SPP
- SPP_DAG

### Measurements

- MPP
- MXX
- MYY
- MZZ

### Resets

- MR
- MRX
- MRY
- R
- RX
- RY

### Noise And Errors

- DEPOLARIZE1
- DEPOLARIZE2
- E
- ELSE_CORRELATED_ERROR
- HERALDED_ERASE
- HERALDED_PAULI_CHANNEL_1
- I_ERROR
- II_ERROR
- PAULI_CHANNEL_1
- PAULI_CHANNEL_2
- X_ERROR
- Y_ERROR
- Z_ERROR
