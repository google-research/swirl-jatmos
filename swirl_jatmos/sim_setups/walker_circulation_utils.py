# Copyright 2024 The swirl_jatmos Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Miscellaneous utilities for the Mock-Walker Circulation simulation."""

import os
from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

Array: TypeAlias = jax.Array

_SOUNDING_FILENAME = 'theta_li_qt_sounding.csv'


def save_sounding(
    output_dir: str, z_c: np.ndarray, theta_li: np.ndarray, q_t: np.ndarray
) -> None:
  """Save a sounding of 1D quantities (θ_li(z) and q_t(z) only).

  This function is meant to be used interactively, e.g., from a colab.

  Args:
    output_dir: The directory to save the sounding to.
    z_c: The z coordinates of the grid.
    theta_li: The liquid-ice potential temperature.
    q_t: The total specific humidity.
  """
  assert z_c.ndim == 1 and theta_li.ndim == 1 and q_t.ndim == 1
  assert len(z_c) == len(theta_li) == len(q_t)
  assert np.all(q_t >= 0.0), 'Negative q_t detected.'
  df = pd.DataFrame({
      'z': z_c,
      'theta_li': theta_li,
      'q_t': q_t,
  })
  path = os.path.join(output_dir, _SOUNDING_FILENAME)
  df.to_csv(path, index=False)


def load_sounding(dirname: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Load sounding of θ_li(z) and q_t(z) from CSV file."""
  path = os.path.join(dirname, _SOUNDING_FILENAME)
  df = pd.read_csv(path)
  z_c = df['z'].to_numpy()
  theta_li = df['theta_li'].to_numpy()
  q_t = df['q_t'].to_numpy()
  return z_c, theta_li, q_t


def load_sounding_and_interpolate(
    dirname: str, z_c: Array
) -> tuple[Array, Array]:
  """Load sounding data for θ_li(z) and q_t(z), and interpolate onto `z_c`.

  Note: Although interpolation is used, the intention is that the sounding is
  provided at the same z coordinates as the grid to be used.  The only
  difference is that the input `z_c` may contain halos and so extrapolation onto
  halo coordinates is used.  However, those halo values are supposed to never be
  used anyway.

  The returned θ_li(z) and q_t(z) are shape (1, 1, nz) JAX arrays.

  Args:
    dirname: The directory containing the CSV file with the sounding.
    z_c: The z coordinates on nodes, shape (1, 1, nz).

  Returns:
    Tuple of (θ_li, q_t) JAX arrays loaded from the sounding, of shape
    (1, 1, nz).
  """
  z_c = jnp.squeeze(z_c)
  z_c_data, theta_li_data, q_t_data = load_sounding(dirname)

  # Interpolate sounding data onto the provided z_c.
  theta_li = jnp.interp(z_c, z_c_data, theta_li_data)
  q_t = jnp.interp(z_c, z_c_data, q_t_data)

  # After interpolating, cast to the desired dtype (dtype of input `z_c`).
  theta_li = jnp.astype(theta_li, z_c.dtype)
  q_t = jnp.astype(q_t, z_c.dtype)

  # Return as (1, 1, nz) arrays.
  theta_li = theta_li[jnp.newaxis, jnp.newaxis, :]
  q_t = q_t[jnp.newaxis, jnp.newaxis, :]
  return theta_li, q_t


def regularize_q_t_for_sounding(
    z: np.ndarray, q_t: np.ndarray, index: int, q_inf: float, s: float
) -> np.ndarray:
  """Regularize q_t for sounding by ensuring q_t > 0.

  Numerical schemes for convection can lead to q_t having negative values, even
  in the horizontal mean.  To create a sounding, we want to ensure that q_t is
  nonnegative.  This function regularizes q_t by replacing values with z-indices
  greater than `index` with the following functional form:

      g(z) = q_∞ + (q_0 - q_∞) * exp[-(z - z0) / s]      z >= z0

  where

      z_0 = z[index]
      q_0 = q_t[index]

  and q_∞, s are free choices for the asymptotic value of q_t and the choice of
  decay rate towards q_∞, respectively.

  Args:
    z: The z coordinates of the grid, shape (1, 1, nz).
    q_t: The total specific humidity, shape (1, 1, nz).
    index: The z-index to use as z_0.
    q_inf: The asymptotic value of q_t.
    s: The choice of decay length towards q_inf.

  Returns:
    The regularized q_t, a 1D numpy array.
  """
  z_0 = z[index]
  q_0 = q_t[index]
  ind = z >= z_0
  q_t_reg = 1.0 * q_t  # Create a copy.
  q_t_reg[ind] = q_inf + (q_0 - q_inf) * np.exp(-(z[ind] - z_0) / s)
  return q_t_reg


