# -*- coding: utf-8 -*-
"""
Exact sectional curvature computation for the Walsh exponential family.

At P_k (boundary of the simplex), the FIM has rank 15 (support = 16 points).
We work in the active 15-dimensional eigenspace of the FIM.

The Riemannian curvature tensor of the Levi-Civita connection for an
exponential family is (Amari & Nagaoka, Ch. 3):
    R_{abcd} = (1/4) sum_e [ T_{ace} (G^{-1})_{ef} T_{bdf}
                              - T_{ade} (G^{-1})_{ef} T_{bcf} ]
where T_{abc} is the third cumulant tensor (third derivative of psi),
G is the FIM restricted to the active subspace, and G^{-1} is its inverse.

Sectional curvature in the plane (e_a, e_b) of the active eigenbasis:
    K(a,b) = R_{abab} / (G_{aa}*G_{bb} - G_{ab}^2)
Since we diagonalise G, G_{ab} = lambda_a * delta_{ab}, so:
    K(a,b) = R_{abab} / (lambda_a * lambda_b)
    R_{abab} = (1/4) sum_c [ T~_{aac} * T~_{bbc} - T~_{abc}^2 ] / lambda_c
"""

import sys, os
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from scipy.linalg import eigh
from collections import Counter

# ── S-boxes ───────────────────────────────────────────────────────────────────
S_PRESENT = [0xC,0x5,0x6,0xB,0x9,0x0,0xA,0xD,0x3,0xE,0xF,0x8,0x4,0x7,0x1,0x2]
S_GIFT    = [0x1,0xA,0x4,0xC,0x6,0xF,0x3,0x9,0x2,0xD,0xB,0x7,0x5,0x0,0x8,0xE]
S_SKINNY  = [0xC,0x6,0x9,0x0,0x1,0xA,0x2,0xB,0x3,0x8,0x5,0xD,0x4,0xE,0x7,0xF]

s = 4
n = 1 << s   # 16

def dot(a, b):
    return bin(a & b).count('1') % 2

def compute_wht_table(S):
    """Full 16x16 WHT table: wht[alpha, beta] = WHT_S(alpha, beta)."""
    W = np.zeros((n, n))
    for alpha in range(n):
        for beta in range(n):
            W[alpha, beta] = sum((-1)**(dot(alpha,a) ^ dot(beta,S[a]))
                                 for a in range(n))
    return W

def sectional_curvatures(S, tol=1e-8):
    """
    Compute all C(r,2) sectional curvatures in the active eigenspace of the
    FIM at P_0.  Returns (K_values, eigenvalues, rank).
    """
    W = compute_wht_table(S)   # 16x16, W[0,0]=16
    w = W / n                  # normalized: w[alpha,beta] = WHT_S/n

    # ── Coordinates ──────────────────────────────────────────────────────────
    coords = [(a,b) for a in range(n) for b in range(n) if not (a==0 and b==0)]
    d = len(coords)            # 255
    alpha_arr = np.array([c[0] for c in coords])
    beta_arr  = np.array([c[1] for c in coords])

    # ── Expectation parameters eta_i = w[alpha_i, beta_i] ───────────────────
    eta = w[alpha_arr, beta_arr]   # shape (d,)

    # ── XOR index arrays (for vectorised lookups) ────────────────────────────
    xA = alpha_arr[:,None] ^ alpha_arr[None,:]   # d×d
    xB = beta_arr[:,None]  ^ beta_arr[None,:]    # d×d

    # ── FIM: F[i,j] = w[ai^aj, bi^bj] - eta[i]*eta[j] ──────────────────────
    F = w[xA, xB] - eta[:,None]*eta[None,:]      # d×d

    # ── Active eigenspace (rank = 15 for s=4) ───────────────────────────────
    eigvals, eigvecs = eigh(F)
    active = eigvals > tol
    lam = eigvals[active]          # r positive eigenvalues
    U   = eigvecs[:, active]       # d×r  (r = rank ≈ 15)
    r   = len(lam)

    # ── "Shifted lookup" for Term 1 of third cumulant ───────────────────────
    # sum_c[c, alpha', beta'] = sum_k U[k,c] * w[(alpha')^ak, (beta')^bk]
    sum_c = np.zeros((r, n, n))
    for c in range(r):
        for k_idx in range(d):
            uk = U[k_idx, c]
            if abs(uk) < 1e-14:
                continue
            ak, bk = coords[k_idx]
            shift_a = np.arange(n) ^ ak
            shift_b = np.arange(n) ^ bk
            sum_c[c] += uk * w[np.ix_(shift_a, shift_b)]
    # T~_1[:,:,c] = U^T @ M_c @ U  where M_c[i,j] = sum_c[c, ai^aj, bi^bj]
    T_tilde = np.zeros((r, r, r))
    for c in range(r):
        M_c = sum_c[c, xA, xB]         # d×d
        T_tilde[:,:,c] = U.T @ M_c @ U  # r×r

    # ── Correction terms ─────────────────────────────────────────────────────
    # W_ij[i,j] = w[ai^aj, bi^bj]  (no subtraction of eta*eta^T)
    Wij      = w[xA, xB]              # d×d
    UtWU     = U.T @ Wij @ U          # r×r  (= projected E[Ti Tj])
    Ut_eta   = U.T @ eta               # r

    # Term 2: -w[ai^aj,bi^bj] * eta[k]
    T_tilde -= np.einsum('ab,c->abc', UtWU, Ut_eta)
    # Term 3: -w[ai^ak,bi^bk] * eta[j]
    T_tilde -= np.einsum('ac,b->abc', UtWU, Ut_eta)
    # Term 4: -w[aj^ak,bj^bk] * eta[i]
    T_tilde -= np.einsum('bc,a->abc', UtWU, Ut_eta)
    # Term 5: +2*eta[i]*eta[j]*eta[k]
    T_tilde += 2.0 * np.einsum('a,b,c->abc', Ut_eta, Ut_eta, Ut_eta)

    # Symmetry check (should be 0 up to floating-point)
    sym_err = np.max(np.abs(T_tilde - T_tilde.transpose(1,0,2)))
    if sym_err > 1e-10:
        print(f"  WARNING: T~ not symmetric: max|T~_abc - T~_bac| = {sym_err:.2e}")

    # ── Sectional curvatures ─────────────────────────────────────────────────
    # T_aa[a,c] = T~[a,a,c]
    T_aa = np.array([T_tilde[a,a,:] for a in range(r)])   # r×r

    inv_lam = 1.0 / lam
    # R_num[a,b] = (1/4)*sum_c (T_aa[a,c]*T_aa[b,c] - T~[a,b,c]^2) / lam[c]
    R_num = (0.25 * (T_aa * inv_lam) @ T_aa.T
             - 0.25 * np.einsum('abc,c->ab', T_tilde**2, inv_lam))

    # K[a,b] = R_num[a,b] / (lam[a]*lam[b])  for a != b
    K_vals = []
    for a in range(r):
        for b in range(a+1, r):
            K_vals.append(R_num[a,b] / (lam[a]*lam[b]))

    return np.array(K_vals), lam, r


# ── Run for the three S-boxes ─────────────────────────────────────────────────
print("=" * 65)
print("SECTIONAL CURVATURES OF E_S IN THE ACTIVE 15-DIMENSIONAL EIGENSPACE")
print("(at P_0; identical for all P_k by the key-translation symmetry)")
print("=" * 65)

results = {}
for name, S in [("PRESENT", S_PRESENT), ("GIFT", S_GIFT), ("SKINNY", S_SKINNY)]:
    print(f"\n--- {name} ---")
    K, lam, r = sectional_curvatures(S)
    print(f"  FIM rank = {r}  (expected 15)")
    print(f"  Eigenvalues (FIM active): min={lam.min():.6f}, max={lam.max():.6f}")
    print(f"  Number of sectional curvature pairs: {len(K)}  (= C({r},2))")
    print(f"  Min  K = {K.min():.6f}")
    print(f"  Max  K = {K.max():.6f}")
    print(f"  Mean K = {K.mean():.6f}")
    print(f"  #(K > 0)  = {(K > 1e-12).sum()}")
    print(f"  #(K = 0)  = {(np.abs(K) <= 1e-12).sum()}")
    print(f"  #(K < 0)  = {(K < -1e-12).sum()}")
    results[name] = K

print("\n" + "="*65)
print("SUMMARY TABLE (for LaTeX)")
print("="*65)
print(f"{'S-box':<10} {'Min K':>10} {'Max K':>10} {'Mean K':>10} "
      f"{'#K>0':>6} {'#K=0':>6} {'#K<0':>6}")
for name, K in results.items():
    print(f"{name:<10} {K.min():>10.6f} {K.max():>10.6f} {K.mean():>10.6f} "
          f"{(K>1e-12).sum():>6} {(np.abs(K)<=1e-12).sum():>6} {(K<-1e-12).sum():>6}")
