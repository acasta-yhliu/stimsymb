from __future__ import annotations

from dataclasses import dataclass

import galois
import numpy as np
from numpy.typing import NDArray
from sympy.logic.boolalg import Boolean, false


GF2Matrix = NDArray[np.uint8]
GF2 = galois.GF(2)


def _symplectic_products(
    left_xs: GF2Matrix,
    left_zs: GF2Matrix,
    right_xs: GF2Matrix,
    right_zs: GF2Matrix,
) -> GF2Matrix:
    """Return pairwise binary symplectic products between Pauli row sets."""
    return (left_xs @ right_zs.T + left_zs @ right_xs.T) % 2


def _check_gf2_matrix(name: str, matrix: GF2Matrix, shape: tuple[int, int]) -> None:
    """Validate that a matrix has the expected GF(2) shape and values."""
    if matrix.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if matrix.dtype != np.uint8:
        raise TypeError(f"{name} must have uint8 dtype")
    if np.any(matrix > 1):
        raise ValueError(f"{name} must contain only 0 or 1")


def _gf2_rank(matrix: GF2Matrix) -> int:
    """Return the row rank of a GF(2) matrix."""
    rref = np.asarray(GF2(matrix).row_reduce(), dtype=np.uint8)
    return int(np.count_nonzero(np.any(rref, axis=1)))


def _gf2_solve(matrix: GF2Matrix, rhs: GF2Matrix) -> GF2Matrix:
    """Return one solution to a GF(2) linear system."""
    rows, cols = matrix.shape
    aug = np.concatenate([matrix.copy(), rhs.reshape(rows, 1)], axis=1)
    pivot_cols: list[int] = []
    pivot_row = 0

    for col in range(cols):
        pivots = np.flatnonzero(aug[pivot_row:, col])
        if len(pivots) == 0:
            continue

        row = pivot_row + int(pivots[0])
        aug[[pivot_row, row]] = aug[[row, pivot_row]]

        # Eliminate this pivot column from every other row over GF(2).
        for other in range(rows):
            if other != pivot_row and aug[other, col]:
                aug[other] ^= aug[pivot_row]

        pivot_cols.append(col)
        pivot_row += 1
        if pivot_row == rows:
            break

    coefficients = aug[:, :cols]
    if np.any((~np.any(coefficients, axis=1)) & (aug[:, cols] == 1)):
        raise ValueError("GF(2) system is inconsistent")

    solution = np.zeros(cols, dtype=np.uint8)
    for row, col in enumerate(pivot_cols):
        solution[col] = aug[row, cols]
    return solution


def _destabilizers_for(
    stabilizer_xs: GF2Matrix,
    stabilizer_zs: GF2Matrix,
) -> tuple[GF2Matrix, GF2Matrix]:
    """Return destabilizer support dual to independent commuting stabilizers."""
    n = stabilizer_xs.shape[0]
    system = np.concatenate([stabilizer_zs, stabilizer_xs], axis=1)
    destabilizer_xs = np.zeros((n, n), dtype=np.uint8)
    destabilizer_zs = np.zeros((n, n), dtype=np.uint8)

    for row in range(n):
        rhs = np.eye(n, dtype=np.uint8)[row]
        solution = _gf2_solve(system, rhs)
        destabilizer_xs[row] = solution[:n]
        destabilizer_zs[row] = solution[n:]

    for i in range(n):
        for j in range(i + 1, n):
            anticommutes = _symplectic_products(
                destabilizer_xs[i : i + 1],
                destabilizer_zs[i : i + 1],
                destabilizer_xs[j : j + 1],
                destabilizer_zs[j : j + 1],
            )[0, 0]
            if anticommutes:
                destabilizer_xs[j] ^= stabilizer_xs[i]
                destabilizer_zs[j] ^= stabilizer_zs[i]

    return destabilizer_xs, destabilizer_zs


@dataclass(slots=True)
class SymbolicTableau:
    """Symbolic stabilizer tableau with concrete GF(2) support and symbolic phases."""

    num_qubits: int
    xs: GF2Matrix
    zs: GF2Matrix
    phases: list[Boolean]

    def __post_init__(self) -> None:
        rows = 2 * self.num_qubits
        shape = (rows, self.num_qubits)
        if self.num_qubits < 0:
            raise ValueError("num_qubits must be non-negative")
        if self.xs.shape != shape or self.zs.shape != shape:
            raise ValueError(f"xs and zs must have shape {shape}")
        if self.xs.dtype != np.uint8 or self.zs.dtype != np.uint8:
            raise TypeError("xs and zs must have uint8 dtype")
        if np.any(self.xs > 1) or np.any(self.zs > 1):
            raise ValueError("xs and zs must contain only 0 or 1")
        if len(self.phases) != rows:
            raise ValueError(f"phases must have length {rows}")

    @property
    def destabilizer_xs(self) -> GF2Matrix:
        return self.xs[: self.num_qubits]

    @property
    def destabilizer_zs(self) -> GF2Matrix:
        return self.zs[: self.num_qubits]

    @property
    def destabilizer_phases(self) -> list[Boolean]:
        return self.phases[: self.num_qubits]

    @property
    def stabilizer_xs(self) -> GF2Matrix:
        return self.xs[self.num_qubits :]

    @property
    def stabilizer_zs(self) -> GF2Matrix:
        return self.zs[self.num_qubits :]

    @property
    def stabilizer_phases(self) -> list[Boolean]:
        return self.phases[self.num_qubits :]

    def satisfy_canonical_commutation(self) -> bool:
        """Return whether destabilizers and stabilizers have canonical commutation."""
        n = self.num_qubits
        eye = np.eye(n, dtype=np.uint8)
        zero = np.zeros((n, n), dtype=np.uint8)

        dd = _symplectic_products(
            self.destabilizer_xs,
            self.destabilizer_zs,
            self.destabilizer_xs,
            self.destabilizer_zs,
        )
        ds = _symplectic_products(
            self.destabilizer_xs,
            self.destabilizer_zs,
            self.stabilizer_xs,
            self.stabilizer_zs,
        )
        ss = _symplectic_products(
            self.stabilizer_xs,
            self.stabilizer_zs,
            self.stabilizer_xs,
            self.stabilizer_zs,
        )
        return (
            np.array_equal(dd, zero)
            and np.array_equal(ds, eye)
            and np.array_equal(ss, zero)
        )

    @classmethod
    def zero_state(cls, num_qubits: int) -> SymbolicTableau:
        """Return the canonical tableau for |0...0>."""
        if num_qubits < 0:
            raise ValueError("num_qubits must be non-negative")

        rows = 2 * num_qubits
        xs = np.zeros((rows, num_qubits), dtype=np.uint8)
        zs = np.zeros((rows, num_qubits), dtype=np.uint8)
        phases = [false] * rows

        for qubit in range(num_qubits):
            xs[qubit, qubit] = 1
            zs[num_qubits + qubit, qubit] = 1

        return cls(
            num_qubits=num_qubits,
            xs=xs,
            zs=zs,
            phases=phases,
        )

    @classmethod
    def from_stabilizers(
        cls,
        xs: GF2Matrix,
        zs: GF2Matrix,
        phases: list[Boolean] | None = None,
    ) -> SymbolicTableau:
        """Return a canonical tableau from independent commuting stabilizers."""
        if xs.shape != zs.shape or xs.ndim != 2 or xs.shape[0] != xs.shape[1]:
            raise ValueError("stabilizer xs and zs must both have shape (n, n)")

        n = xs.shape[0]
        _check_gf2_matrix("stabilizer xs", xs, (n, n))
        _check_gf2_matrix("stabilizer zs", zs, (n, n))

        stabilizer_phases = [false] * n if phases is None else phases
        if len(stabilizer_phases) != n:
            raise ValueError(f"stabilizer phases must have length {n}")

        if np.any(_symplectic_products(xs, zs, xs, zs)):
            raise ValueError("stabilizers must commute")
        if _gf2_rank(np.concatenate([xs, zs], axis=1)) != n:
            raise ValueError("stabilizers must be independent")

        destabilizer_xs, destabilizer_zs = _destabilizers_for(xs, zs)
        return cls(
            num_qubits=n,
            xs=np.concatenate([destabilizer_xs, xs], axis=0),
            zs=np.concatenate([destabilizer_zs, zs], axis=0),
            phases=[false] * n + stabilizer_phases,
        )
