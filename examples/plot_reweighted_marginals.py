#!/usr/bin/env python3
"""Demonstrate get_weights by reweighting a uniform-in-sin²θ prior across parameterisations.

This script:
- Samples uniformly in sin²θ12, sin²θ23, sin²θ13 ∈ [0, 1] and δ ∈ (−π, π] under e3.
- Transforms each sample to each of the nine parameterisations and computes importance
  weights using get_weights.
- For each target, plots the weighted (blue) and unweighted (orange) marginals of
  sin²θ12, sin²θ23, sin²θ13, and δ.
- Flat weighted histograms confirm that the weights correctly map the e3 uniform prior
  to a uniform prior in the target parameterisation.

Run:
    python examples/plot_reweighted_marginals.py
    Output saved to figures/reweighted_marginals.png
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from pmns_transforms.core import transform, get_weights  # noqa: E402

ORIGIN = 'e3'
PARAMS = ['e1', 'e2', 'e3', 'mu1', 'mu2', 'mu3', 'tau1', 'tau2', 'tau3']

# Labels from the convention in arXiv:2507.02101
LABEL_MAP = {
    'e1': r'$\nu_\mu\nu_\tau/\nu_2\nu_3$',
    'e2': r'$\nu_\mu\nu_\tau/\nu_1\nu_3$',
    'e3': r'$\nu_\mu\nu_\tau/\nu_1\nu_2$',
    'mu1': r'$\nu_e\nu_\tau/\nu_2\nu_3$',
    'mu2': r'$\nu_e\nu_\tau/\nu_1\nu_3$',
    'mu3': r'$\nu_e\nu_\tau/\nu_1\nu_2$',
    'tau1': r'$\nu_e\nu_\mu/\nu_2\nu_3$',
    'tau2': r'$\nu_e\nu_\mu/\nu_1\nu_3$',
    'tau3': r'$\nu_e\nu_\mu/\nu_1\nu_2$',
}

VAR_LABELS = [
    r'$\sin^2\theta_{12}$',
    r'$\sin^2\theta_{23}$',
    r'$\sin^2\theta_{13}$',
    r'$\delta_{CP}$',
]


def sample_uniform_sinsq(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    sin2_th12 = rng.uniform(0.0, 1.0, size=n)
    sin2_th23 = rng.uniform(0.0, 1.0, size=n)
    sin2_th13 = rng.uniform(0.0, 1.0, size=n)
    dcp = rng.uniform(-np.pi, np.pi, size=n)
    th12 = np.arcsin(np.sqrt(sin2_th12))
    th23 = np.arcsin(np.sqrt(sin2_th23))
    th13 = np.arcsin(np.sqrt(sin2_th13))
    return th12, th23, th13, dcp


def plot_reweighted_marginals(n: int = 1_000_000, seed: int = 0, bins: int = 100,
                               outpath: str | None = None):
    if outpath is None:
        outpath = str(ROOT / 'figures' / 'reweighted_marginals.png')
    out = pathlib.Path(outpath)
    out.parent.mkdir(parents=True, exist_ok=True)

    th12, th23, th13, dcp = sample_uniform_sinsq(n, seed)

    fig = plt.figure(figsize=(16, 18))
    fig.suptitle(
        rf'Marginal $\sin^2\theta$ distributions — uniform in {ORIGIN}, reweighted to each target'
        '\n'
        r'(flat blue $\Rightarrow$ weights correct;  orange = unweighted)',
        fontsize=12,
    )
    outer = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.3)

    for idx, tgt in enumerate(PARAMS):
        new_th12, new_th23, new_th13, new_dcp = transform(ORIGIN, tgt, th12, th23, th13, dcp)
        weights = get_weights(ORIGIN, tgt, th12, th23, th13, dcp)

        params_arr = np.array([
            np.sin(new_th12) ** 2,
            np.sin(new_th23) ** 2,
            np.sin(new_th13) ** 2,
            new_dcp,
        ])
        valid = np.isfinite(weights) & np.all(np.isfinite(params_arr), axis=0)
        w = weights[valid]
        p = params_arr[:, valid]
        w_norm = w / w.sum()

        row, col = divmod(idx, 3)
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer[row, col], hspace=0.5, wspace=0.4
        )

        for k, (arr, label) in enumerate(zip(p, VAR_LABELS)):
            ax = fig.add_subplot(inner[k // 2, k % 2])
            lo, hi = arr.min(), arr.max()
            bin_width = (hi - lo) / bins

            # Normalise to probability density: integral over [lo, hi] = 1
            bc_w, edges = np.histogram(arr, bins=bins, range=(lo, hi), weights=w_norm)
            bc_w = bc_w / bin_width
            bc_u, _ = np.histogram(arr, bins=bins, range=(lo, hi))
            bc_u = bc_u / (bc_u.sum() * bin_width)
            centres = 0.5 * (edges[:-1] + edges[1:])
            flat_level = 1.0 / (hi - lo)

            ax.step(centres, bc_w, where='mid', color='tab:blue', lw=1.2, label='weighted')
            ax.step(centres, bc_u, where='mid', color='tab:orange', lw=0.8, alpha=0.7, label='unweighted')
            ax.axhline(flat_level, color='k', ls='--', lw=0.8, alpha=0.5)
            ax.set_ylim(bottom=0)
            ax.set_title(label, fontsize=7, pad=2)
            ax.tick_params(axis='both', labelsize=6)
            if k == 0:
                ax.legend(fontsize=5, loc='upper right', framealpha=0.5)

        title_ax = fig.add_subplot(outer[row, col])
        title_ax.set_title(f'{ORIGIN} → {tgt}  {LABEL_MAP[tgt]}', fontsize=9, pad=14)
        title_ax.axis('off')

    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved to {out}')


if __name__ == '__main__':
    plot_reweighted_marginals()
