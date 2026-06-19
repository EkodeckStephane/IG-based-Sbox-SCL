# -*- coding: utf-8 -*-
"""Sanity-check the active curvature diagnostic on two 4-bit bijections."""

import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from scipy.linalg import eigh

S_PRESENT = [0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD,
             0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2]
S_RANDOM = [3, 14, 1, 10, 6, 12, 11, 0, 9, 7, 4, 2, 5, 8, 15, 13]

s = 4
n = 1 << s


def dot(a, b):
    return bin(a & b).count("1") % 2


def fim_and_curvatures(S):
    W = np.array(
        [[sum((-1) ** (dot(al, a) ^ dot(be, S[a])) for a in range(n))
          for be in range(n)] for al in range(n)],
        dtype=float,
    )
    w = W / n
    coords = [(a, b) for a in range(n) for b in range(n) if (a, b) != (0, 0)]
    aA = np.array([c[0] for c in coords])
    bA = np.array([c[1] for c in coords])
    eta = w[aA, bA]
    xA = aA[:, None] ^ aA[None, :]
    xB = bA[:, None] ^ bA[None, :]
    F = w[xA, xB] - eta[:, None] * eta[None, :]
    lam, U = eigh(F)
    active = lam > 1e-8
    lam = lam[active]
    U = U[:, active]
    r = len(lam)

    print(f"  Eigenvalues: min={lam.min():.6f}, max={lam.max():.6f}, n={n}")
    print(f"  Rank={r}, all lam==n? {np.allclose(lam, n)}")

    sum_c = np.zeros((r, n, n))
    for c in range(r):
        for k, (ak, bk) in enumerate(coords):
            uk = U[k, c]
            if abs(uk) < 1e-14:
                continue
            sa = np.arange(n) ^ ak
            sb = np.arange(n) ^ bk
            sum_c[c] += uk * w[np.ix_(sa, sb)]

    T = np.zeros((r, r, r))
    for c in range(r):
        M = sum_c[c, xA, xB]
        T[:, :, c] = U.T @ M @ U

    Wij = w[xA, xB]
    UtWU = U.T @ Wij @ U
    Ue = U.T @ eta
    T -= np.einsum("ab,c->abc", UtWU, Ue)
    T -= np.einsum("ac,b->abc", UtWU, Ue)
    T -= np.einsum("bc,a->abc", UtWU, Ue)
    T += 2 * np.einsum("a,b,c->abc", Ue, Ue, Ue)

    inv_lam = 1 / lam
    T_aa = np.array([T[a, a, :] for a in range(r)])
    R_num = (
        0.25 * (T_aa * inv_lam) @ T_aa.T
        - 0.25 * np.einsum("abc,c->ab", T ** 2, inv_lam)
    )
    K = np.array(
        [R_num[a, b] / (lam[a] * lam[b]) for a in range(r) for b in range(a + 1, r)]
    )
    print(f"  K: min={K.min():.8f}, max={K.max():.8f}")
    print(f"  All K == -1/4? {np.allclose(K, -0.25)}")
    return K


print("=" * 55)
print("ACTIVE CURVATURE DIAGNOSTIC SANITY CHECK")
print("=" * 55)

for name, S in [("PRESENT", S_PRESENT), ("Random bijection", S_RANDOM)]:
    print(f"\n--- {name} ---")
    fim_and_curvatures(S)

print("\n" + "=" * 55)
print("INTERPRETATION")
print("=" * 55)
print(
    """
This script verifies the computational protocol used in the paper on PRESENT
and on one fixed random 4-bit bijection.

1. The active covariance rank/eigenvalue statement is proved in the paper:
   for a bijective 4-bit S-box, the active dimension is 15 and the active
   eigenvalues equal 16 in the unnormalised Walsh basis.

2. The curvature value K = -1/4 is reported as a reproducible diagnostic
   under the projection, normalization, and sign conventions implemented here.

3. A complete symbolic characterization of the curvature diagnostic is left
   as an open problem in the manuscript.
"""
)
