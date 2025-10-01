# pmns-transforms

A lightweight scientific Python library to build, transform, and analyse PMNS mixing matrices across nine Tait–Bryan parameterisations that privilege specific flavour/mass symmetries.

This codebase is based on and aligned with the framework developed in **Testing T2K’s Bayesian constraints with priors in alternate parameterisations (arXiv:2507.02101).**

The library implements a numerically stable, vectorised API to:
- construct a PMNS matrix for any of the nine parameterisations;
- extract the mixing angles and Dirac CP phase from a PMNS matrix under a target parameterisation;
- transform parameters from one parameterisation to another (composition of the above two);
- compute the Jarlskog invariant.

## Parameterisation overview

- Parameterisations are identified by strings in {e, mu, tau}{1,2,3}, for example: e3 (canonical/PDG-like), mu1, tau2, etc.
- Each string indicates the “single element” index (the one equal to s13 e^{-i δ}) and thereby fixes the row/column symmetries privileged by that parameterisation (as in the paper’s promoted row–column symmetry).
- **See appendix A of arXiv:2507.02101 to identify parameterisations with the corresponding row/column symmetries and their matrix form.**

## Conventions and ranges

- All angles are in radians.
- Angle ranges: θ12, θ23, θ13 ∈ [0, π/2]; δ ∈ (−π, π].
- Single-element convention: U_single = s13 · exp(−i δ) at the position set by the chosen parameterisation.
- Broadcasting: all public functions accept scalars or array-like inputs and broadcast them with NumPy semantics. Scalars yield 3×3 arrays; arrays yield outputs with leading (3,3) followed by the broadcast shape.

### Non-identifiable regimes and NaN policy

Certain inverse mappings (matrix → angles) become singular or numerically ill-conditioned:
- When c13 ≈ 0, θ12 and θ23 are not identifiable under this chart; they are returned as NaN.
- When the denominator for cos δ (2 s12 c12 s23 c23 s13) is ill-conditioned, δ is returned as NaN.
- When sin δ ≈ 0, the sign of δ is resolved deterministically via cos δ: δ = 0 if cos δ ≥ 0, else δ = π.

This mirrors the parameterisation singularities discussed in the paper and yields stable, deterministic behaviour.

## Installation

Requirements
- Python ≥ 3.8
- NumPy ≥ 1.21

Install from source
- Standard install:
  - `pip install .`
- Editable install (development):
  - `pip install -e .`

## Quick start

```python
import numpy as np
from pmns_transforms.core import get_mixing_matrix, get_parameters, transform, get_Jarlskog

# Example angles (radians)
th12, th23, th13 = 0.59, 0.79, 0.15
dcp = 0.30

# 1) Build the PMNS matrix in the canonical e3 parameterisation
U = get_mixing_matrix('e3', th12, th23, th13, dcp)  # shape (3, 3)

# 2) Extract angles under a different parameterisation (e.g., mu1)
#    original_parameterisation informs the phase/sign convention of U
rec_th12, rec_th23, rec_th13, rec_dcp = get_parameters('mu1', U, original_parameterisation='e3')

# 3) Transform angles directly between parameterisations
new_th12, new_th23, new_th13, new_dcp = transform('e3', 'mu1', th12, th23, th13, dcp)

# 4) Compute the Jarlskog invariant (broadcasted over inputs)
Jcp = get_Jarlskog(th12, th23, th13, dcp)
```

### Broadcasting example
```python
# Vectorised inputs broadcast automatically
th12 = np.linspace(0.2, 1.3, 1000)
th23 = 0.80
th13 = 0.15
dcp  = np.linspace(-np.pi, np.pi, 1000)

U_all = get_mixing_matrix('e2', th12, th23, th13, dcp)  # shape (3, 3, 1000)
```

## API summary

- get_mixing_matrix(parameterisation, th12, th23, th13, dcp) → ndarray
  - Builds U under the chosen parameterisation. Returns (3,3) or (3,3,*broadcast_shape).

- get_parameters(target_parameterisation, mixing_matrix, original_parameterisation='e3') → (th12, th23, th13, dcp)
  - Extracts angles/phase under target_parameterisation, assuming mixing_matrix was built using original_parameterisation. Handles non-identifiable regimes via NaN, and sign choices as documented below.

- transform(original_parameterisation, target_parameterisation, th12, th23, th13, dcp) → (new_th12, new_th23, new_th13, new_dcp)
  - Convenience wrapper: build → extract.

- get_Jarlskog(th12, th23, th13, dcp) → ndarray
  - Computes J = s12 c12 s23 c23 s13 c13² sin δ. Returns a real array matching the broadcast shape.


## Numerical notes

- Inverse stability:
  - The sine and cosine of the angles are clipped for numerical safety before square roots/acos.
  - If cos(θ13) is below a small epsilon: θ12 and θ23 → NaN.
  - If the denominator of the expression for cos δ is ill-conditioned: δ → NaN.

- Shape checks: get_parameters coerces the input matrix to an ndarray and validates a leading 3×3 shape.

## Citing

If you use this package in your work, please cite following paper:
- Testing T2K’s Bayesian constraints with priors in alternate parameterisations: arXiv:2507.02101.

## License

GPL-3.0-or-later. See LICENSE for details.
