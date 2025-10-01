#!/usr/bin/env python3
"""Overlay distributions over the square sines of the angles after transforming into the standard parameterisation from all other parameterisations.

This script:
- Samples uniformly in s12^2, s23^2, s13^2 ∈ [0, 1] and δ ∈ (−π, π].
- Treats the sampled angles as belonging to each of the eight non-standard parameterisations, 
  transforms them to the standard, and overlays histograms for comparison.
- Plots sin^2(θ12), sin^2(θ23), sin^2(θ13), δ, and Jarlskog invariant Jcp.
- Reproduces (at a rough level) Figure 6 of arXiv:2507.02101

Run:
- python scripts/plot_taitBryanPriors.py
- Output saved to .overlaid_taitBryanPriors.png
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt

# Import package from src
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from pmns_transforms.core import transform, get_Jarlskog

SOURCES = ['e1', 'e2', 'mu1', 'mu2', 'mu3', 'tau1', 'tau2', 'tau3']

# Labels from the convention in arxiv:2507.02101
LABEL_MAP = {
    'e1': r'$\nu_\mu\nu_\tau/\nu_2\nu_3$',
    'e2': r'$\nu_\mu\nu_\tau/\nu_1\nu_3$',
    'e3': r'$\nu_\mu\nu_\tau/\nu_1\nu_2$',
    'mu1': r'$\nu_e\nu_\tau/\nu_2\nu_3$',
    'mu2': r'$\nu_e\nu_\tau/\nu_1\nu_3$',
    'mu3': r'$\nu_e\nu_\tau/\nu_1\nu_2$',
    'tau1': r'$\nu_e\nu_\mu/\nu_2\nu_3$',
    'tau2': r'$\nu_e\nu_\mu/\nu_1\nu_3$',
    'tau3': r'$\nu_e\nu_\mu/\nu_1\nu_2$'
}

def sample_uniform_sinsq(n: int, seed: int = 0, eps: float = 1e-6):
    rng = np.random.default_rng(seed)
    # Avoid exact 0 and 1 to limit non-identifiability
    u12 = rng.uniform(0.0, 1.0, size=n)
    u23 = rng.uniform(0.0, 1.0, size=n)
    u13 = rng.uniform(0.0, 1.0, size=n)
    s12sq = eps + (1.0 - 2.0 * eps) * u12
    s23sq = eps + (1.0 - 2.0 * eps) * u23
    s13sq = eps + (1.0 - 2.0 * eps) * u13
    dcp = rng.uniform(-np.pi, np.pi, size=n)

    th12 = np.arcsin(np.sqrt(s12sq))
    th23 = np.arcsin(np.sqrt(s23sq))
    th13 = np.arcsin(np.sqrt(s13sq))
    return th12, th23, th13, dcp


def _sinsq(x: np.ndarray) -> np.ndarray:
    return np.sin(x) ** 2


def plot_overlaid_distributions_to_e3(n: int = 1_000_000, seed: int = 0, outpath: str = 'dists_overlaid_to_standard.png'):
    out = pathlib.Path(outpath)
    out.parent.mkdir(parents=True, exist_ok=True)

    th12, th23, th13, dcp = sample_uniform_sinsq(n, seed)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    ax_s12, ax_s23, ax_s13, ax_d, ax_j, ax_unused = axes.ravel()

    # Bins
    bins_s = np.linspace(0.0, 1.0, 100)
    bins_d = np.linspace(-np.pi, np.pi, 100)
    bins_j = np.linspace(-0.12, 0.12, 101)

    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', None)
    family_ls = {'e': '-', 'mu': '--', 'tau': ':'}

    for idx, src in enumerate(SOURCES):
        color = None if colors is None else colors[idx % len(colors)]
        family = 'e' if src.startswith('e') else ('mu' if src.startswith('mu') else 'tau')
        ls = family_ls[family]
        lw = 1.8 if src in {'mu1', 'tau2'} else 1.2

        new_th12, new_th23, new_th13, new_dcp = transform(src, 'e3', th12, th23, th13, dcp)

        s12sq_new = _sinsq(new_th12)
        s23sq_new = _sinsq(new_th23)
        s13sq_new = _sinsq(new_th13)

        jcp = get_Jarlskog(new_th12, new_th23, new_th13, new_dcp)

        m12 = ~np.isnan(s12sq_new)
        m23 = ~np.isnan(s23sq_new)
        m13 = ~np.isnan(s13sq_new)
        md = ~np.isnan(new_dcp)
        mj = ~np.isnan(jcp)

        label = LABEL_MAP[src]

        ax_s12.hist(s12sq_new[m12], bins=bins_s, histtype='step', density=True, label=label, color=color, alpha=0.95, linestyle=ls, linewidth=lw)
        ax_s23.hist(s23sq_new[m23], bins=bins_s, histtype='step', density=True, label=label, color=color, alpha=0.95, linestyle=ls, linewidth=lw)
        ax_s13.hist(s13sq_new[m13], bins=bins_s, histtype='step', density=True, label=label, color=color, alpha=0.95, linestyle=ls, linewidth=lw)
        ax_d.hist(new_dcp[md], bins=bins_d, histtype='step', density=True, label=label, color=color, alpha=0.95, linestyle=ls, linewidth=lw)
        ax_j.hist(jcp[mj], bins=bins_j, histtype='step', density=True, label=label, color=color, alpha=0.95, linestyle=ls, linewidth=lw)

    # Labels and titles
    ax_s12.set_title('sin²(θ12) in e3')
    ax_s23.set_title('sin²(θ23) in e3')
    ax_s13.set_title('sin²(θ13) in e3')
    ax_d.set_title('δ [rad] in e3')
    ax_j.set_title('Jarlskog Jcp in e3')

    for ax in [ax_s12, ax_s23, ax_s13]:
        ax.set_xlim(0, 1)
    ax_d.set_xlim(-np.pi, np.pi)
    ax_j.set_xlim(bins_j[0], bins_j[-1])

    for ax in [ax_s12, ax_s23, ax_s13, ax_d, ax_j]:
        ax.grid(True, alpha=0.3)

    ax_unused.axis('off')
    handles, labels = ax_s12.get_legend_handles_labels()
    ax_unused.legend(handles, labels, loc='center', ncol=2, fontsize='small', frameon=False, title='Source parameterisations')

    fig.suptitle('Overlaid distributions after transforming the standard parameterisation (e3)', y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    plot_overlaid_distributions_to_e3()
