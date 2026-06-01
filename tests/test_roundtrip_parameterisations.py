"""Round-trip and edge-case tests for all 9×9 parameterisation pairs and public API error paths."""

import numpy as np
import pytest

from pmns_transforms.core import get_parameters, get_mixing_matrix, get_Jarlskog, transform


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
