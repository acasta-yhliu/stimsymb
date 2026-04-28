# Stim Instruction Support TODO

## Supported Now

- All single-qubit Clifford gates
- REPEAT

## Single-Qubit Clifford Gates

- I
- X
- Y
- Z
- H
- S
- S_DAG
- SQRT_X
- SQRT_X_DAG
- SQRT_Y
- SQRT_Y_DAG
- H_XY
- H_YZ
- H_NXY
- H_NXZ
- H_NYZ
- C_XYZ
- C_ZYX
- C_NXYZ
- C_NZYX
- C_XNYZ
- C_XYNZ
- C_ZNYX
- C_ZYNX

## Two-Qubit Clifford Gates

- II
- CX
- CY
- CZ
- XCX
- XCY
- XCZ
- YCX
- YCY
- YCZ
- SWAP
- CXSWAP
- CZSWAP
- SWAPCX
- ISWAP
- ISWAP_DAG
- SQRT_XX
- SQRT_XX_DAG
- SQRT_YY
- SQRT_YY_DAG
- SQRT_ZZ
- SQRT_ZZ_DAG

## Multi-Pauli Unitaries

- SPP
- SPP_DAG

## Measurements

- M
- MX
- MY
- MPP
- MXX
- MYY
- MZZ
- MPAD

## Resets

- R
- RX
- RY
- MR
- MRX
- MRY

## Metadata And Structure

- REPEAT
- TICK
- QUBIT_COORDS
- SHIFT_COORDS
- DETECTOR
- OBSERVABLE_INCLUDE

## Noise And Errors

- X_ERROR
- Y_ERROR
- Z_ERROR
- I_ERROR
- II_ERROR
- PAULI_CHANNEL_1
- PAULI_CHANNEL_2
- DEPOLARIZE1
- DEPOLARIZE2
- E
- ELSE_CORRELATED_ERROR
- HERALDED_ERASE
- HERALDED_PAULI_CHANNEL_1
