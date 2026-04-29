from __future__ import annotations

from dataclasses import dataclass

import galois
import numpy as np
from numpy.typing import NDArray
from sympy.logic.boolalg import Boolean, Xor, false, true


GF2Matrix = NDArray[np.uint8]
GF2 = galois.GF(2)


@dataclass(slots=True)
class CanonicalForm:
    """Canonical stabilizer rows used for symbolic state equality."""

    xs: GF2Matrix
    zs: GF2Matrix
    phases: tuple[Boolean, ...]

    def __eq__(self, other: object) -> bool:
        """Return whether two canonical forms have identical Pauli support."""
        if not isinstance(other, CanonicalForm):
            return NotImplemented
        return np.array_equal(self.xs, other.xs) and np.array_equal(self.zs, other.zs)


def _symplectic_products(
    left_xs: GF2Matrix,
    left_zs: GF2Matrix,
    right_xs: GF2Matrix,
    right_zs: GF2Matrix,
) -> GF2Matrix:
    """Return pairwise binary symplectic products between Pauli row sets."""
    return (left_xs @ right_zs.T + left_zs @ right_xs.T) % 2


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
    def canonical_form(self) -> CanonicalForm:
        """Return the canonical stabilizer form for equality checking."""
        n = self.num_qubits
        return _canonicalize_stabilizers(
            self.xs[n:],
            self.zs[n:],
            self.phases[n:],
        )

    def satisfy_canonical_commutation(self) -> bool:
        """Return whether destabilizers and stabilizers have canonical commutation."""
        n = self.num_qubits
        eye = np.eye(n, dtype=np.uint8)
        zero = np.zeros((n, n), dtype=np.uint8)
        destabilizer_xs = self.xs[:n]
        destabilizer_zs = self.zs[:n]
        stabilizer_xs = self.xs[n:]
        stabilizer_zs = self.zs[n:]

        dd = _symplectic_products(
            destabilizer_xs,
            destabilizer_zs,
            destabilizer_xs,
            destabilizer_zs,
        )
        ds = _symplectic_products(
            destabilizer_xs,
            destabilizer_zs,
            stabilizer_xs,
            stabilizer_zs,
        )
        ss = _symplectic_products(
            stabilizer_xs,
            stabilizer_zs,
            stabilizer_xs,
            stabilizer_zs,
        )
        return (
            np.array_equal(dd, zero)
            and np.array_equal(ds, eye)
            and np.array_equal(ss, zero)
        )

    def multiply_row(self, target: int, source: int) -> None:
        """Right-multiply one tableau row by another commuting tableau row."""
        rows = 2 * self.num_qubits
        if target < 0 or target >= rows:
            raise IndexError("target row index out of range")
        if source < 0 or source >= rows:
            raise IndexError("source row index out of range")

        anticommutes = _symplectic_products(
            self.xs[target : target + 1],
            self.zs[target : target + 1],
            self.xs[source : source + 1],
            self.zs[source : source + 1],
        )[0, 0]
        if anticommutes:
            raise ValueError("cannot multiply anticommuting tableau rows")

        # Row multiplication preserves the represented stabilizer group while
        # changing the tableau basis.
        zx = int(self.zs[target] @ self.xs[source])
        xz = int(self.xs[target] @ self.zs[source])
        self.xs[target] ^= self.xs[source]
        self.zs[target] ^= self.zs[source]
        self.phases[target] = Xor(
            self.phases[target],
            self.phases[source],
            true if (zx - xz) % 4 == 2 else false,
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
        if int(np.linalg.matrix_rank(GF2(np.concatenate([xs, zs], axis=1)))) != n:
            raise ValueError("stabilizers must be independent")

        # Destabilizers are a dual basis: D_i anticommutes only with S_i.
        destabilizer_xs, destabilizer_zs = _destabilizers_for(xs, zs)
        return cls(
            num_qubits=n,
            xs=np.concatenate([destabilizer_xs, xs], axis=0),
            zs=np.concatenate([destabilizer_zs, zs], axis=0),
            phases=[false] * n + stabilizer_phases,
        )


def _canonicalize_stabilizers(
    xs: GF2Matrix,
    zs: GF2Matrix,
    phases: list[Boolean],
) -> CanonicalForm:
    """Return row-reduced stabilizers with matching symbolic phases."""
    xs = xs.copy()
    zs = zs.copy()
    phases = phases.copy()
    rows, num_qubits = xs.shape
    support = np.concatenate([xs, zs], axis=1)
    pivot_row = 0

    for col in range(2 * num_qubits):
        pivots = np.flatnonzero(support[pivot_row:, col])
        if len(pivots) == 0:
            continue

        source = pivot_row + int(pivots[0])
        if source != pivot_row:
            support[[pivot_row, source]] = support[[source, pivot_row]]
            xs[[pivot_row, source]] = xs[[source, pivot_row]]
            zs[[pivot_row, source]] = zs[[source, pivot_row]]
            phases[pivot_row], phases[source] = phases[source], phases[pivot_row]

        for row in range(rows):
            if row == pivot_row or not support[row, col]:
                continue

            # RREF row addition is Pauli-row multiplication. Since stabilizers
            # commute, support XOR is enough, with this sign correction.
            zx = int(zs[row] @ xs[pivot_row])
            xz = int(xs[row] @ zs[pivot_row])
            xs[row] ^= xs[pivot_row]
            zs[row] ^= zs[pivot_row]
            support[row] ^= support[pivot_row]
            phases[row] = Xor(
                phases[row],
                phases[pivot_row],
                true if (zx - xz) % 4 == 2 else false,
            )

        pivot_row += 1
        if pivot_row == rows:
            break

    return CanonicalForm(
        xs=xs,
        zs=zs,
        phases=tuple(phases),
    )


def _destabilizers_for(
    stabilizer_xs: GF2Matrix,
    stabilizer_zs: GF2Matrix,
) -> tuple[GF2Matrix, GF2Matrix]:
    """Return destabilizer support dual to independent commuting stabilizers."""
    n = stabilizer_xs.shape[0]

    # For each destabilizer D_i = (x | z), require <D_i, S_j> = delta_ij.
    # Since <(x | z), (sx | sz)> = x.sz + z.sx over GF(2), this is the
    # linear system [S_z | S_x] [x | z]^T = e_i.
    system = np.concatenate([stabilizer_zs, stabilizer_xs], axis=1)
    destabilizer_xs = np.zeros((n, n), dtype=np.uint8)
    destabilizer_zs = np.zeros((n, n), dtype=np.uint8)

    for row in range(n):
        # Solving against e_i gives a Pauli that anticommutes with S_i and
        # commutes with every other stabilizer. Free variables are set to 0.
        rhs = np.eye(n, dtype=np.uint8)[row]
        solution = _gf2_solve(system, rhs)
        destabilizer_xs[row] = solution[:n]
        destabilizer_zs[row] = solution[n:]

    # The dual-basis equations do not force the destabilizers to commute with
    # each other. Triangular cleanup fixes D_i/D_j anticommutation without
    # changing any D_j/S_k relation.
    for i in range(n):
        for j in range(i + 1, n):
            # Stabilizer row i is the canonical partner of destabilizer row i.
            # Multiplying it into row j removes D_i/D_j anticommutation.
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


def _check_gf2_matrix(name: str, matrix: GF2Matrix, shape: tuple[int, int]) -> None:
    """Validate that a matrix has the expected GF(2) shape and values."""
    if matrix.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if matrix.dtype != np.uint8:
        raise TypeError(f"{name} must have uint8 dtype")
    if np.any(matrix > 1):
        raise ValueError(f"{name} must contain only 0 or 1")


def _gf2_solve(matrix: GF2Matrix, rhs: GF2Matrix) -> GF2Matrix:
    """Return one solution to a GF(2) linear system."""
    rows, cols = matrix.shape
    aug = np.concatenate([matrix.copy(), rhs.reshape(rows, 1)], axis=1)
    rref = np.asarray(GF2(aug).row_reduce(), dtype=np.uint8)

    coefficients = rref[:, :cols]
    if np.any((~np.any(coefficients, axis=1)) & (rref[:, cols] == 1)):
        raise ValueError("GF(2) system is inconsistent")

    solution = np.zeros(cols, dtype=np.uint8)
    for row in range(rows):
        pivot = np.flatnonzero(coefficients[row])
        if len(pivot):
            solution[int(pivot[0])] = rref[row, cols]
    return solution
