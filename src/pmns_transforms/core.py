"""pmns_transforms.core

Core routines to build and manipulate PMNS mixing matrices under different
parameterisations and to extract angles and the Dirac CP phase.

Conventions
- Angles are in radians.
- The "single element" convention is U_single = s13 * exp(-i * dcp) at the
  index implied by the chosen parameterisation.
- All public functions support NumPy broadcasting. Scalar inputs yield
  scalar outputs (or a 3x3 matrix), array-like inputs yield outputs with the
  broadcasted shape.

Notes
- Parameterisations are encoded as strings in {e, mu, tau}{1,2,3}, e.g.,
  "e3", "mu1", "tau2"; Greek letters μ, τ are normalised to "mu", "tau".
- The extraction of dcp follows the sign convention of the original
  parameterisation used to build the matrix; this is why both the original and
  target parameterisations are required when transforming/extracting.
- When c13 is effectively zero, get_parameters returns θ12 and θ23 as NaN to
  indicate non-identifiability under this parameterisation. When the cos δ
  denominator is ill-conditioned, δ is returned as NaN to indicate the same.
"""

import re
import numpy as np
import numpy.typing as npt
from typing import Tuple

# Single indexes of parameterisations (position of the simple element in U_PMNS). The labels correspond
# to the promoted row-column symmetry as given in https://arxiv.org/abs/2507.02101

# e1 ----> "label":r'$\nu_\mu\nu_\tau/\nu_2\nu_3$'
# e2 ----> "label":r'$\nu_\mu\nu_\tau/\nu_1\nu_3$'
# e3 ----> "label":r'$\nu_\mu\nu_\tau/\nu_1\nu_2$'

# mu1 ---> "label":r'$\nu_e\nu_\tau/\nu_2\nu_3$'
# mu2 ---> "label":r'$\nu_e\nu_\tau/\nu_1\nu_3$'
# mu3 ---> "label":r'$\nu_e\nu_\tau/\nu_1\nu_2$'

# tau1 --> "label":r'$\nu_e\nu_\mu/\nu_2\nu_3$'
# tau2 --> "label":r'$\nu_e\nu_\mu/\nu_1\nu_3$'
# tau3 --> "label":r'$\nu_e\nu_\mu/\nu_1\nu_2$'


def _set_indices(alpha, i):
    """Resolve index tuples associated with a parameterisation.

    This is a low-level helper that maps a parameterisation identified by
    (alpha, i), where alpha ∈ {"e", "mu", "tau"} and i ∈ {1, 2, 3}, to the
    positions in the 3×3 PMNS matrix corresponding to:
    - the "single" element (s13 e^{-iδ}),
    - the two elements that depend on θ12,
    - the two elements that depend on θ23,
    - and the four "compound" elements that mix the angles and δ.

    Parameters
    ----------
    alpha : str
        One of {"e", "mu", "tau"}.
    i : int
        One of {1, 2, 3}.

    Returns
    -------
    tuple
        A 4-tuple:
        - single_index : tuple[int, int]
        - th12_indices : list[tuple[int, int]] of length 2
        - th23_indices : list[tuple[int, int]] of length 2
        - compound_indices : list[tuple[int, int]] of length 4

    Raises
    ------
    ValueError
        If ``alpha`` is not one of {"e", "mu", "tau"} or ``i`` not in {1, 2, 3}.
    """
    index_mapping = {"e": 1, "mu": 2, "tau": 3}

    if alpha not in index_mapping:
        raise ValueError("alpha must be one of 'e', 'mu', 'tau'")
    if i not in (1, 2, 3):
        raise ValueError("i must be one of {1, 2, 3}")

    alpha_idx = index_mapping[alpha]

    a = alpha_idx - 1
    j = i - 1
    not_a = list(range(a)) + list(range(a + 1, 3))
    not_j = list(range(j)) + list(range(j + 1, 3))

    single_index = (a, j)
    th12_indices = [(a, not_j[0]), (a, not_j[1])]
    th23_indices = [(not_a[0], j), (not_a[1], j)]
    compound_indices = [
        (not_a[0], not_j[0]),
        (not_a[0], not_j[1]),
        (not_a[1], not_j[0]),
        (not_a[1], not_j[1]),
    ]

    return single_index, th12_indices, th23_indices, compound_indices


def _parse_parameterisation_string(param_str):
    """Parse a parameterisation string into (alpha, i).

    The accepted format is one of: "e1", "e2", "e3", "mu1", "mu2", "mu3",
    "tau1", "tau2", "tau3". Case-insensitive; spaces and underscores are
    ignored. Greek letters are normalised: μ → "mu", τ → "tau".

    Parameters
    ----------
    param_str : str
        The parameterisation string to parse.

    Returns
    -------
    tuple[str, int]
        A pair (alpha, i) with alpha ∈ {"e", "mu", "tau"} and i ∈ {1, 2, 3}.

    Raises
    ------
    TypeError
        If ``param_str`` is not a string.
    ValueError
        If ``param_str`` does not match a valid parameterisation.
    """
    if not isinstance(param_str, str):
        raise TypeError("Parameterisation string must be of type str")

    s = param_str.strip().lower()
    # Normalize some common variations
    s = s.replace("μ", "mu").replace("τ", "tau")
    s = s.replace(" ", "").replace("_", "")

    m = re.fullmatch(r"(e|mu|tau)([123])", s)
    if not m:
        raise ValueError(
            f"Invalid parameterisation string: {param_str!r}. Expected one of e1,e2,e3,mu1,mu2,mu3,tau1,tau2,tau3."
        )
    alpha = m.group(1)
    i = int(m.group(2))
    return alpha, i


def get_mixing_matrix(parameterisation: str, th12: npt.ArrayLike, th23: npt.ArrayLike, th13: npt.ArrayLike, dcp: npt.ArrayLike) -> np.ndarray:
    """Build a PMNS mixing matrix for a given parameterisation.

    Angles are in radians. Inputs can be scalars or array-like; all inputs are
    broadcast to a common shape using NumPy broadcasting. The returned array has
    shape (3, 3) for scalar inputs, or (3, 3, *broadcast_shape) for array
    inputs.

    Parameters
    ----------
    parameterisation : str
        One of {"e1", "e2", "e3", "mu1", "mu2", "mu3", "tau1", "tau2", "tau3"}.
        Determines the position of the simple element s13 e^{-iδ}.
    th12, th23, th13 : array-like
        Mixing angles θ12, θ23, θ13 in radians with values in [0, π/2]. Scalars
        or broadcastable arrays are supported.
    dcp : array-like
        Dirac CP phase δ in radians with values in (−π, π]. Scalars or
        broadcastable arrays are supported.

    Returns
    -------
    numpy.ndarray
        Complex array of shape (3, 3) or (3, 3, *broadcast_shape) containing the
        PMNS matrix/matrices under the chosen parameterisation.

    Raises
    ------
    ValueError
        If ``parameterisation`` is invalid.
    """
    parsed_par = _parse_parameterisation_string(parameterisation)

    # Broadcast all inputs to a common shape
    th12_arr, th23_arr, th13_arr, dcp_arr = np.broadcast_arrays(
        np.asarray(th12), np.asarray(th23), np.asarray(th13), np.asarray(dcp)
    )

    s12 = np.sin(th12_arr)
    c12 = np.cos(th12_arr)
    s23 = np.sin(th23_arr)
    c23 = np.cos(th23_arr)
    s13 = np.sin(th13_arr)
    c13 = np.cos(th13_arr)

    e_minus_i_dcp = np.exp(-1j * dcp_arr)
    e_plus_i_dcp = np.exp(1j * dcp_arr)

    single_element = s13 * e_minus_i_dcp
    th12_elements = np.array([c12 * c13, s12 * c13])
    th23_elements = np.array([s23 * c13, c23 * c13])
    compound_elements = np.array([
        -s12 * c23 - c12 * s23 * s13 * e_plus_i_dcp,
        +c12 * c23 - s12 * s23 * s13 * e_plus_i_dcp,
        +s12 * s23 - c12 * c23 * s13 * e_plus_i_dcp,
        -c12 * s23 - s12 * c23 * s13 * e_plus_i_dcp,
    ])

    broadcast_shape = th12_arr.shape
    mixing_shape = (3, 3) + broadcast_shape
    mixing_matrix = np.zeros(mixing_shape, dtype=np.complex128)

    single_index, th12_indices, th23_indices, compound_indices = _set_indices(parsed_par[0], parsed_par[1])

    mixing_matrix[single_index] = single_element
    rows, cols = zip(*th12_indices)
    mixing_matrix[rows, cols] = th12_elements
    rows, cols = zip(*th23_indices)
    mixing_matrix[rows, cols] = th23_elements
    rows, cols = zip(*compound_indices)
    mixing_matrix[rows, cols] = compound_elements

    # If scalar inputs (no broadcasted dims), return (3,3)
    if broadcast_shape == ():
        return mixing_matrix.reshape(3, 3)

    return mixing_matrix


def get_parameters(target_parameterisation: str, mixing_matrix: npt.ArrayLike, original_parameterisation: str = 'e3') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract (θ12, θ23, θ13, δ) from a PMNS matrix under a target parameterisation.

    The extraction assumes the matrix was generated using the given
    ``original_parameterisation``, which fixes the sign convention for the
    single element U_single = s13 e^{-iδ}. This is needed because the PMNS
    matrix is degenerate under certain phase/sign redefinitions.

    Edge-case behaviour:
    - If c13 ≈ 0 (within a small epsilon), θ12 and θ23 cannot be identified from
      the entries used here. In this case they are returned as NaN to signal
      non-identifiability under this parameterisation.
    - If the denominator in the cos δ expression is ill-conditioned, δ is
      returned as NaN to signal non-identifiability.

    Parameters
    ----------
    target_parameterisation : str
        Desired parameterisation for the returned angles; one of {"e1","e2",
        "e3","mu1","mu2","mu3","tau1","tau2","tau3"}.
    mixing_matrix : array-like
        Complex array with shape (3, 3) or (3, 3, *extra_dims). A list of lists
        is also accepted and will be coerced to a NumPy array.
    original_parameterisation : str, optional
        Parameterisation that was used to build ``mixing_matrix``; defaults to
        "e3".

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        (th12, th23, th13, dcp), each with shape matching the broadcasted shape
        of ``mixing_matrix`` (or scalars when the input matrix is 2-D).

    Raises
    ------
    ValueError
        If either parameterisation string is invalid.
    """

    mixing_matrix = np.asarray(mixing_matrix)
    if mixing_matrix.ndim < 2 or mixing_matrix.shape[0] != 3 or mixing_matrix.shape[1] != 3:
        raise ValueError(
            f"mixing_matrix must have shape (3, 3, ...) with leading 3x3, got shape {mixing_matrix.shape}"
        )
    parsed_original_par = _parse_parameterisation_string(original_parameterisation)
    parsed_target_par = _parse_parameterisation_string(target_parameterisation)

    single_index, th12_indices, th23_indices, compound_indices = _set_indices(parsed_target_par[0], parsed_target_par[1])
    old_single_index = _set_indices(parsed_original_par[0], parsed_original_par[1])[0]

    # Extract magnitudes and compute sines/cosines with numerical safety
    eps = 1e-12
    eps2 = 1e-14

    imPart = np.asanyarray(mixing_matrix[old_single_index]).imag

    s13 = np.abs(mixing_matrix[single_index])
    s13 = np.clip(s13, 0.0, 1.0)
    c13 = np.sqrt(np.clip(1.0 - s13 * s13, 0.0, 1.0))

    # Safe division by c13 for s12 and s23
    c13_safe = np.where(c13 > eps, c13, 1.0)

    s12_num = np.abs(mixing_matrix[th12_indices[1]])
    s12 = np.where(c13 > eps, s12_num / c13_safe, np.nan)
    s12 = np.clip(s12, 0.0, 1.0)
    c12 = np.sqrt(np.clip(1.0 - s12 * s12, 0.0, 1.0))

    s23_num = np.abs(mixing_matrix[th23_indices[0]])
    s23 = np.where(c13 > eps, s23_num / c13_safe, np.nan)
    s23 = np.clip(s23, 0.0, 1.0)
    c23 = np.sqrt(np.clip(1.0 - s23 * s23, 0.0, 1.0))

    # cos(delta) from the chosen compound element
    compound_element_sq = np.abs(mixing_matrix[compound_indices[-1]]) ** 2
    term1 = c12 ** 2 * s23 ** 2
    term2 = (s12 ** 2) * (c23 ** 2) * (s13 ** 2)
    denom = 2.0 * s12 * c12 * s23 * c23 * s13

    valid = denom > eps
    denom_safe = np.where(valid, denom, 1.0)

    cos_dcp = (compound_element_sq - term1 - term2) / denom_safe
    cos_dcp = np.clip(cos_dcp, -1.0, 1.0)

    delta_mag = np.arccos(cos_dcp)

    # Determine delta: if sin δ ≈ 0, choose 0 or π from cos δ; otherwise use sign from Im(U_single).
    sign_im = np.sign(imPart)
    signless = np.abs(imPart) <= eps2

    dcp_valid = np.where(
        signless,
        np.where(cos_dcp < 0.0, np.pi, 0.0),
        -sign_im * delta_mag,
    )

    dcp = np.where(valid, dcp_valid, np.nan)

    th12 = np.arcsin(s12)
    th23 = np.arcsin(s23)
    th13 = np.arcsin(s13)

    return th12, th23, th13, dcp


def transform(original_parameterisation: str, target_parameterisation: str, th12: npt.ArrayLike, th23: npt.ArrayLike, th13: npt.ArrayLike, dcp: npt.ArrayLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert angles from one parameterisation to another.

    This is a convenience wrapper around ``get_mixing_matrix`` followed by
    ``get_parameters`` using the appropriate original/target parameterisations.

    Parameters
    ----------
    original_parameterisation : str
        Parameterisation of the input angles.
    target_parameterisation : str
        Desired parameterisation for the output angles.
    th12, th23, th13 : array-like
        Input mixing angles in radians with values in [0, π/2]. Scalars or
        broadcastable arrays are supported.
    dcp : array-like
        Input Dirac CP phase δ in radians with values in (−π, π]. Scalars or
        broadcastable arrays are supported.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        (new_th12, new_th23, new_th13, new_dcp), matching the broadcasted shape
        of the inputs (or scalars for scalar inputs).
    """

    mixing_matrix = get_mixing_matrix(original_parameterisation, th12, th23, th13, dcp)
    new_th12, new_th23, new_th13, new_dcp = get_parameters(
        target_parameterisation, mixing_matrix, original_parameterisation
    )

    return new_th12, new_th23, new_th13, new_dcp


def get_Jarlskog(th12: npt.ArrayLike, th23: npt.ArrayLike, th13: npt.ArrayLike, dcp: npt.ArrayLike) -> np.ndarray:
    """Compute the Jarlskog invariant Jcp.

    Implements J = s12 c12 s23 c23 s13 (1 - s13^2) sin δ for any
    parameterisation. Inputs can be scalars or array-like and will be broadcast
    together.

    Parameters
    ----------
    th12, th23, th13 : array-like
        Mixing angles in radians with values in [0, π/2].
    dcp : array-like
        Dirac CP phase δ in radians with values in (−π, π].

    Returns
    -------
    numpy.ndarray
        Real array with the broadcasted shape of the inputs.
    """
    s12 = np.sin(th12)
    s23 = np.sin(th23)
    s13 = np.sin(th13)

    c12 = np.sqrt(1 - s12 ** 2)
    c23 = np.sqrt(1 - s23 ** 2)
    c13sqr = 1 - s13 ** 2

    Jcp = s12 * s23 * s13 * c12 * c23 * c13sqr * np.sin(dcp)

    return Jcp
