"""Round-trip and edge-case tests for all 9×9 parameterisation pairs and public API error paths."""

import numpy as np
import pytest

from pmns_transforms.core import get_parameters, get_mixing_matrix, get_Jarlskog, transform, get_jacobian, get_weights


PARAMS = ['e1', 'e2', 'e3', 'mu1', 'mu2', 'mu3', 'tau1', 'tau2', 'tau3']


def _wrap_angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return minimal signed difference a-b wrapped to (-pi, pi]."""
    d = a - b
    d = (d + np.pi) % (2 * np.pi) - np.pi
    # Map -pi to pi for consistent comparison window (optional)
    d = np.where(np.isclose(d, -np.pi), np.pi, d)
    return d


@pytest.mark.parametrize("seed", [0, 1])
def test_roundtrip_parameterisations(seed):
    rng = np.random.default_rng(seed)

    # Generate angles away from singular/ill-conditioned boundaries
    # th12, th23 in [0.2, 1.37] (~[11.5°, 78.5°])
    # th13 in [0.1, 1.2] (~[5.7°, 68.8°]) ensures c13 not too small
    # dcp in (-pi+1e-2, pi-1e-2)
    def sample(n):
        th12 = rng.uniform(0.2, 1.37, size=n)
        th23 = rng.uniform(0.2, 1.37, size=n)
        th13 = rng.uniform(0.1, 1.2, size=n)
        dcp = rng.uniform(-np.pi + 1e-2, np.pi - 1e-2, size=n)
        return th12, th23, th13, dcp

    # Test both scalar and vectorised cases
    cases = [1, 5]

    for n in cases:
        th12, th23, th13, dcp = sample(n)
        # Cast to scalar when n == 1
        if n == 1:
            th12, th23, th13, dcp = th12.item(), th23.item(), th13.item(), dcp.item()

        for orig in PARAMS:
            for tgt in PARAMS:
                new_th12, new_th23, new_th13, new_dcp = transform(orig, tgt, th12, th23, th13, dcp)
                back_th12, back_th23, back_th13, back_dcp = transform(tgt, orig, new_th12, new_th23, new_th13, new_dcp)

                # Convert to arrays for uniform handling
                th12_a = np.asarray(th12)
                th23_a = np.asarray(th23)
                th13_a = np.asarray(th13)
                dcp_a = np.asarray(dcp)

                back_th12 = np.asarray(back_th12)
                back_th23 = np.asarray(back_th23)
                back_th13 = np.asarray(back_th13)
                back_dcp = np.asarray(back_dcp)

                # Assert no NaNs in the safe region
                assert not np.isnan(back_th12).any()
                assert not np.isnan(back_th23).any()
                assert not np.isnan(back_th13).any()
                assert not np.isnan(back_dcp).any()

                # Numerical tolerances
                atol = 5e-10
                rtol = 5e-10

                assert np.allclose(back_th12, th12_a, rtol=rtol, atol=atol)
                assert np.allclose(back_th23, th23_a, rtol=rtol, atol=atol)
                assert np.allclose(back_th13, th13_a, rtol=rtol, atol=atol)

                d = _wrap_angle_diff(back_dcp, dcp_a)
                assert np.allclose(d, 0.0, rtol=rtol, atol=1e-9)


@pytest.mark.parametrize("param", PARAMS)
def test_nan_at_c13_zero(param):
    """When th13=π/2 (c13≈0), th12 and th23 are non-identifiable and must be NaN."""
    U = get_mixing_matrix(param, th12=0.3, th23=0.7, th13=np.pi / 2, dcp=1.2)
    th12, th23, th13, dcp = get_parameters(param, U, original_parameterisation=param)
    assert np.isnan(th12), f"{param}: expected th12=NaN when c13≈0"
    assert np.isnan(th23), f"{param}: expected th23=NaN when c13≈0"


def test_jarlskog_known_value():
    """Spot-check against the analytic value for symmetric angles."""
    # All angles = π/4, δ = π/2  →  J = (1/√2)^5 · (1/2) · 1 = 1/(8√2)
    J = get_Jarlskog(np.pi / 4, np.pi / 4, np.pi / 4, np.pi / 2)
    assert np.isclose(J, 1.0 / (8.0 * np.sqrt(2)), rtol=1e-12)


def test_jarlskog_broadcasting():
    """get_Jarlskog must broadcast correctly over array inputs."""
    th = np.linspace(0.1, 1.0, 7)
    J = get_Jarlskog(th, th, th, np.pi / 2)
    assert J.shape == (7,)
    assert not np.isnan(J).any()


def test_invalid_parameterisation_raises():
    """Invalid parameterisation strings must raise ValueError or TypeError."""
    U = get_mixing_matrix("e3", 0.3, 0.7, 0.15, 1.2)
    with pytest.raises(ValueError):
        get_mixing_matrix("x9", 0.3, 0.7, 0.15, 1.2)
    with pytest.raises(ValueError):
        get_parameters("bad", U)
    with pytest.raises(TypeError):
        get_parameters(42, U)


def test_jacobian_shape_scalar():
    """get_jacobian returns (4, 4) for scalar inputs."""
    J = get_jacobian("e3", "mu1", 0.58, 0.78, 0.15, 1.2)
    assert J.shape == (4, 4)


def test_jacobian_shape_array():
    """get_jacobian returns (4, 4, N) for 1-D array inputs."""
    n = 5
    th12 = np.full(n, 0.58)
    th23 = np.full(n, 0.78)
    th13 = np.full(n, 0.15)
    dcp = np.full(n, 1.2)
    J = get_jacobian("e3", "mu1", th12, th23, th13, dcp)
    assert J.shape == (4, 4, n)


def test_jacobian_identity_transform():
    """Jacobian of same-to-same parameterisation is the 4×4 identity."""
    for param in PARAMS:
        J = get_jacobian(param, param, 0.58, 0.78, 0.15, 1.2)
        assert np.allclose(J, np.eye(4), atol=1e-7), f"{param}: expected identity Jacobian"


@pytest.mark.parametrize("orig,tgt", [("e3", "mu1"), ("e1", "tau3"), ("mu2", "tau2")])
def test_jacobian_inverse_consistency(orig, tgt):
    """J(A→B) @ J(B→A) ≈ I at a regular interior point."""
    th12, th23, th13, dcp = 0.58, 0.78, 0.15, 1.2
    J_fwd = get_jacobian(orig, tgt, th12, th23, th13, dcp)
    # Evaluate the reverse Jacobian at the transformed point
    new_th12, new_th23, new_th13, new_dcp = transform(orig, tgt, th12, th23, th13, dcp)
    J_rev = get_jacobian(tgt, orig, new_th12, new_th23, new_th13, new_dcp)
    assert np.allclose(J_fwd @ J_rev, np.eye(4), atol=1e-5), (
        f"{orig}→{tgt}: J_fwd @ J_rev not close to identity"
    )


def test_jacobian_det_nonzero():
    """Jacobian determinant is nonzero at an interior point."""
    J = get_jacobian("e3", "mu1", 0.58, 0.78, 0.15, 1.2)
    assert abs(np.linalg.det(J)) > 1e-6


def test_weights_scalar():
    """get_weights returns a positive scalar for scalar inputs."""
    w = get_weights("e3", "mu1", 0.58, 0.78, 0.15, 1.2)
    assert np.ndim(w) == 0
    assert w > 0


def test_weights_array():
    """get_weights returns a 1-D array of positive values for array inputs."""
    n = 5
    th = np.full(n, 0.58)
    w = get_weights("e3", "mu1", th, np.full(n, 0.78), np.full(n, 0.15), np.full(n, 1.2))
    assert w.shape == (n,)
    assert np.all(w > 0)


def test_weights_reciprocal():
    """Weights of A→B and B→A at corresponding points are reciprocals."""
    th12, th23, th13, dcp = 0.58, 0.78, 0.15, 1.2
    w_fwd = get_weights("e3", "mu1", th12, th23, th13, dcp)
    new_th12, new_th23, new_th13, new_dcp = transform("e3", "mu1", th12, th23, th13, dcp)
    w_rev = get_weights("mu1", "e3", new_th12, new_th23, new_th13, new_dcp)
    assert np.isclose(w_fwd, 1.0 / w_rev, rtol=1e-5)


@pytest.mark.parametrize("tgt", PARAMS)
def test_weights_flatten_distribution(tgt):
    """Uniform-in-sin²θ samples in e3, reweighted by get_weights, should be flat in each tgt sin²θ.

    Samples drawn uniformly in (sin²θ12, sin²θ23, sin²θ13, δ) over approximately
    the full physical domain ensure the image in any target parameterisation is
    also the full domain in sin²θ space, making 1D marginals uniform after
    reweighting.  Uses a Pearson chi-squared statistic on weighted histograms
    normalised to the effective sample size N_eff = (Σw)²/Σ(w²).
    """
    rng = np.random.default_rng(0)
    N = 1_000_000
    K = 10

    # Draw uniformly in sin²θ over the full physical domain.
    # Upper bound for sin²θ13 set below sin²(π/2)=1 to avoid the c13≈0 singularity.
    sin2_th12 = rng.uniform(0.0, 1.0, N)
    sin2_th23 = rng.uniform(0.0, 1.0, N)
    sin2_th13 = rng.uniform(0.0, 1.0, N)
    dcp = rng.uniform(-np.pi, np.pi, N)

    th12 = np.arcsin(np.sqrt(sin2_th12))
    th23 = np.arcsin(np.sqrt(sin2_th23))
    th13 = np.arcsin(np.sqrt(sin2_th13))

    new_th12, new_th23, new_th13, new_dcp = transform("e3", tgt, th12, th23, th13, dcp)
    weights = get_weights("e3", tgt, th12, th23, th13, dcp)

    # Histogram in sin²θ space (the natural uniform variable)
    params_arr = np.array([
        np.sin(new_th12) ** 2,
        np.sin(new_th23) ** 2,
        np.sin(new_th13) ** 2,
        new_dcp,
    ])
    valid = np.isfinite(weights) & np.all(np.isfinite(params_arr), axis=0)
    weights = weights[valid]
    params_arr = params_arr[:, valid]

    n_eff = weights.sum() ** 2 / (weights ** 2).sum()
    w_norm = weights * n_eff / weights.sum()

    # With N=10⁶ and correct weights chi2 is well below 100.
    # Wrong weights (1/|J| or constant) give chi2 >> 10,000.
    threshold = 100
    for param_arr, name in zip(params_arr, ("sin²θ12", "sin²θ23", "sin²θ13", "dcp")):
        lo, hi = param_arr.min(), param_arr.max()
        bin_counts, _ = np.histogram(param_arr, bins=K, range=(lo, hi), weights=w_norm)
        expected = n_eff / K
        chi2 = np.sum((bin_counts - expected) ** 2 / expected)
        assert chi2 < threshold, (
            f"e3->{tgt}: {name} not flat after reweighting, chi2={chi2:.1f}"
        )


@pytest.mark.plot
def test_weights_plot_marginals(tmp_path):
    """Generate a diagnostic 9-panel figure: weighted vs unweighted sin²θ histograms.

    Each panel shows the 4 marginals (sin²θ12, sin²θ23, sin²θ13, δ) for one
    target parameterisation.  Blue (weighted) should be flat; orange (unweighted)
    shows the raw density for comparison.  Figure is saved to figures/ in the
    repo root.
    """
    mpl = pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import os

    N = 1_000_000
    K = 50

    rng = np.random.default_rng(0)
    sin2_th12 = rng.uniform(0.0, 1.0, N)
    sin2_th23 = rng.uniform(0.0, 1.0, N)
    sin2_th13 = rng.uniform(0.0, 1.0, N)
    dcp_in = rng.uniform(-np.pi, np.pi, N)
    th12_in = np.arcsin(np.sqrt(sin2_th12))
    th23_in = np.arcsin(np.sqrt(sin2_th23))
    th13_in = np.arcsin(np.sqrt(sin2_th13))

    var_labels = [r"$\sin^2\theta_{12}$", r"$\sin^2\theta_{23}$", r"$\sin^2\theta_{13}$", r"$\delta_{CP}$"]

    fig = plt.figure(figsize=(16, 18))
    fig.suptitle(
        r"Marginal $\sin^2\theta$ distributions — uniform in e3, reweighted to each target"
        "\n"
        r"(blue=weighted should be flat; orange=unweighted for comparison)",
        fontsize=11,
    )
    outer = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.3)

    for idx, tgt in enumerate(PARAMS):
        new_th12, new_th23, new_th13, new_dcp = transform("e3", tgt, th12_in, th23_in, th13_in, dcp_in)
        weights = get_weights("e3", tgt, th12_in, th23_in, th13_in, dcp_in)

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
        inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[row, col], hspace=0.5, wspace=0.4)

        for k, (arr, label) in enumerate(zip(p, var_labels)):
            ax = fig.add_subplot(inner[k // 2, k % 2])
            lo, hi = arr.min(), arr.max()
            bin_width = (hi - lo) / K
            # Normalise to probability density: integral over [lo, hi] = 1
            bc_w, edges = np.histogram(arr, bins=K, range=(lo, hi), weights=w_norm)
            bc_w = bc_w / bin_width
            bc_u, _ = np.histogram(arr, bins=K, range=(lo, hi))
            bc_u = bc_u / (bc_u.sum() * bin_width)
            centres = 0.5 * (edges[:-1] + edges[1:])
            flat_level = 1.0 / (hi - lo)
            ax.step(centres, bc_w, where="mid", color="tab:blue", lw=1.2, label="weighted")
            ax.step(centres, bc_u, where="mid", color="tab:orange", lw=0.8, alpha=0.7, label="unweighted")
            ax.axhline(flat_level, color="k", ls="--", lw=0.8, alpha=0.5)
            ax.set_ylim(bottom=0)
            ax.set_title(label, fontsize=7, pad=2)
            ax.tick_params(axis="both", labelsize=6)
            if k == 0:
                ax.legend(fontsize=5, loc="upper right", framealpha=0.5)

        title_ax = fig.add_subplot(outer[row, col])
        title_ax.set_title(f"e3 → {tgt}", fontsize=9, pad=14)
        title_ax.axis("off")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "reweighted_marginals.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    assert os.path.exists(out_path)


def test_invalid_matrix_shape_raises():
    """Non-3×3 matrix input to get_parameters must raise ValueError."""
    with pytest.raises(ValueError):
        get_parameters("e3", np.ones((2, 2)))


@pytest.mark.parametrize("dcp_boundary", [-np.pi, np.pi])
def test_dcp_boundary(dcp_boundary):
    """dcp at ±π must round-trip without NaN and recover within tolerance."""
    th12_in, th23_in, th13_in = 0.58, 0.78, 0.15
    U = get_mixing_matrix("e3", th12_in, th23_in, th13_in, dcp_boundary)
    th12, th23, th13, dcp = get_parameters("e3", U, original_parameterisation="e3")
    assert not np.isnan(dcp), "dcp=±π should not produce NaN"
    d = _wrap_angle_diff(np.asarray(dcp), np.asarray(dcp_boundary))
    assert np.allclose(d, 0.0, atol=1e-9)
