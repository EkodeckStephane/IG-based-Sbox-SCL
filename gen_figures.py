# -*- coding: utf-8 -*-
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
from fractions import Fraction
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import Counter

S_PRESENT = [0xC,0x5,0x6,0xB,0x9,0x0,0xA,0xD,0x3,0xE,0xF,0x8,0x4,0x7,0x1,0x2]
S_GIFT    = [0x1,0xA,0x4,0xC,0x6,0xF,0x3,0x9,0x2,0xD,0xB,0x7,0x5,0x0,0x8,0xE]
S_SKINNY  = [0xC,0x6,0x9,0x0,0x1,0xA,0x2,0xB,0x3,0x8,0x5,0xD,0x4,0xE,0x7,0xF]

s = 4
n = 16
SBOXES = [("PRESENT", S_PRESENT), ("GIFT", S_GIFT), ("SKINNY", S_SKINNY)]

outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(outdir, exist_ok=True)

def dot(a, b):
    return bin(a & b).count('1') % 2

def wht(S, alpha, beta):
    return sum((-1)**(dot(alpha,a) ^ dot(beta,S[a])) for a in range(n))

def fr_dist_sq_1bit(S, bit_i):
    e_i = 1 << bit_i
    total = Fraction(0)
    for alpha in range(n):
        if not dot(alpha, e_i): continue
        for beta in range(n):
            if alpha == 0 and beta == 0: continue
            w = wht(S, alpha, beta)
            if w == 0: continue
            num = Fraction(4*w*w, n*n)
            den = 1 - Fraction(w*w, n*n)
            if den <= 0: continue
            total += num / den
    return total / n

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
})

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Walsh spectrum heatmaps
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
for ax, (name, S) in zip(axes, SBOXES):
    mat = np.array([[wht(S, a, b) for b in range(n)] for a in range(n)], dtype=float)
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-8, vmax=8, aspect='equal')
    ax.set_title(name, fontweight='bold')
    ax.set_xlabel(r'output mask $\beta$')
    ax.set_ylabel(r'input mask $\alpha$' if ax is axes[0] else '')
    ax.set_xticks([0, 4, 8, 12, 15])
    ax.set_yticks([0, 4, 8, 12, 15])
cb = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02)
cb.set_label(r'$\widehat{S}(\alpha,\beta)$')
fig.suptitle(
    r'Walsh--Hadamard spectra $\widehat{S}(\alpha,\beta)$'
    '\n(red = +8, white = 0, blue = −8; actual max = 8 for all three)',
    fontsize=10)
fig.savefig(os.path.join(outdir, 'fig1_wht_spectra.pdf'), bbox_inches='tight')
plt.close()
print("fig1_wht_spectra.pdf saved")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: FIM diagonal distribution (histogram of |WHT| values)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
bar_labels = ['|WHT|=0\n(F=1)', '|WHT|=4\n(F=15/16)', '|WHT|=8\n(F=3/4)']
colors = ['#2196F3', '#FF9800', '#F44336']
for ax, (name, S) in zip(axes, SBOXES):
    counts = []
    for w_target in [0, 4, 8]:
        c = sum(1 for alpha in range(n) for beta in range(n)
                if (alpha,beta)!=(0,0) and abs(wht(S,alpha,beta))==w_target)
        counts.append(c)
    bars = ax.bar(range(3), counts, color=colors, edgecolor='white', linewidth=0.8)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+2,
                str(cnt), ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(3))
    ax.set_xticklabels(bar_labels, fontsize=8)
    ax.set_ylim(0, 145)
    ax.set_title(name, fontweight='bold')
    ax.set_ylabel('Count' if ax is axes[0] else '')
fig.suptitle(
    r'Distribution of $|\widehat{S}(\alpha,\beta)|$ values'
    '\n(determines FIM diagonal entries; identical for all three S-boxes)',
    fontsize=10)
fig.savefig(os.path.join(outdir, 'fig2_wht_histogram.pdf'), bbox_inches='tight')
plt.close()
print("fig2_wht_histogram.pdf saved")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: FR distance matrix (first-order approx, all 120 pairs)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
for ax, (name, S) in zip(axes, SBOXES):
    mat = np.zeros((n, n))
    # precompute per-bit distances
    per_bit = [float(fr_dist_sq_1bit(S, bit)) for bit in range(s)]
    for k in range(n):
        for kp in range(n):
            if k == kp: continue
            delta = k ^ kp
            mat[k, kp] = sum(per_bit[bit] for bit in range(s) if delta & (1<<bit))
    im = ax.imshow(mat, cmap='plasma', aspect='equal', vmin=0)
    ax.set_title(name, fontweight='bold')
    ax.set_xlabel(r"Key $k^{\prime}$")
    ax.set_ylabel(r"Key $k$" if ax is axes[0] else '')
    ax.set_xticks([0,4,8,12,15]); ax.set_yticks([0,4,8,12,15])
    fig.colorbar(im, ax=ax, shrink=0.85,
                 label=r"$d_F^2$" if ax is axes[2] else '')
fig.suptitle(
    r'First-order Fisher--Rao distance$^2$ between key hypotheses'
    r' (additive in Hamming weight of $k\oplus k^{\prime}$)',
    fontsize=10)
fig.savefig(os.path.join(outdir, 'fig3_fr_distance_matrix.pdf'), bbox_inches='tight')
plt.close()
print("fig3_fr_distance_matrix.pdf saved")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Per-trace Fisher information vs. SNR
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
snr = np.logspace(-2, 3, 400)
line_styles = ['-', '--', ':']
for (name, S), ls in zip(SBOXES, line_styles):
    hws = [bin(S[a]).count('1') for a in range(n)]
    mean_hw = sum(hws) / n
    var_hw = sum((h - mean_hw)**2 for h in hws) / n   # = 1 for all three
    ax.loglog(snr, var_hw * snr, ls=ls, lw=2, label=f"{name} (Var$={var_hw:.0f}$)")
ax.set_xlabel(r'SNR $= 1/\sigma^2$')
ax.set_ylabel(r'Per-trace information $I_S(\sigma) = \mathrm{Var}[\mathrm{HW}(S)]/\sigma^2$')
ax.set_title('Per-trace Fisher information (Hamming weight model)')
ax.legend()
ax.grid(True, which='both', ls='--', alpha=0.3)
# Annotate the unity SNR point
ax.axvline(1, color='grey', ls=':', lw=1)
ax.text(1.1, 0.6, 'SNR=1\n$I_S=1$', fontsize=8, color='grey')
fig.tight_layout()
fig.savefig(os.path.join(outdir, 'fig4_per_trace_info.pdf'), bbox_inches='tight')
plt.close()
print("fig4_per_trace_info.pdf saved")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: FR^2 vs Hamming weight of key difference
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
markers = ['o', 's', '^']
for (name, S), mk in zip(SBOXES, markers):
    per_bit = [float(fr_dist_sq_1bit(S, bit)) for bit in range(s)]
    # Average over all deltas with given HW
    hw_to_avg = {}
    for delta in range(1, n):
        hw = bin(delta).count('1')
        d2 = sum(per_bit[bit] for bit in range(s) if delta & (1<<bit))
        hw_to_avg.setdefault(hw, []).append(d2)
    hws_list = sorted(hw_to_avg)
    avgs = [np.mean(hw_to_avg[hw]) for hw in hws_list]
    mins = [np.min(hw_to_avg[hw]) for hw in hws_list]
    maxs = [np.max(hw_to_avg[hw]) for hw in hws_list]
    ax.plot(hws_list, avgs, mk+'-', lw=1.5, ms=6, label=name)
    ax.fill_between(hws_list, mins, maxs, alpha=0.15)
ax.set_xlabel(r'Hamming weight of $k \oplus k^{\prime}$')
ax.set_ylabel(r'First-order $d_F^2(P_k, P_{k^{\prime}})$')
ax.set_title('Fisher--Rao distance vs. key Hamming distance\n'
             r'(mean $\pm$ range over all deltas of given weight)')
ax.set_xticks([1,2,3,4])
ax.legend()
ax.grid(True, ls='--', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(outdir, 'fig5_fr_vs_hw.pdf'), bbox_inches='tight')
plt.close()
print("fig5_fr_vs_hw.pdf saved")

print("\nAll figures generated in", outdir)
