# Swirl-Jatmos: JAX Atmospheric Simulation
*This is not an officially supported Google product.*

Jatmos is a tool for performing 3D atmospheric large-eddy simulations in a
distributed setting on accelerators such as Google's Tensor Processing Units
(TPUs) and GPUs. Jatmos solves the anelastic Navier-Stokes equations on a
staggered, Arakawa C-grid, with physics modules supporting equilibrium
thermodynamics, one-moment microphysics, and RRTMGP radiative transfer.

Jatmos uses JAX's automatic parallelization to achieve parallelism without the
need to explicitly write communication directives.

## Installation
It is recommended to install in a virtual environment.

```shell
git clone https://github.com/google-research/swirl-jatmos.git
python3 -m pip install -e swirl-jatmos
```

## Demos
See colab demos:

- [Supercell demo](swirl_jatmos/demos/supercell_demo.ipynb)
- [RCEMIP demo](swirl_jatmos/demos/rcemip_demo.ipynb)

## Equations solved
Prognostic equations are solved for the following variables:

- Velocities $u$, $v$, $w$ (anelastic momentum equation)
- Linearized liquid-ice potential temperature $\theta_{li}$
- Total specific humidity $q_t$
- Mass fractions for 2 precipitation species, rain $q_r$ and snow $q_s$

Additionally, the continuity equation is imposed in the form of a
divergence-free mass flux $\nabla \cdot (\rho \mathbf{u}) = 0$, which determines
the pressure $p$ through the solution to a Poisson equation.

## Features of Jatmos

- Staggered grid in Cartesian coordinates, nonuniform grid allowed
- Conservative finite-volume formulation
- WENO5 (among other) methods for convection
- RK3 timestepper
- Adaptive timestepping based on a CFL condition
- Equilibrium thermodynamics for water phases
- One-moment microphysics
- Poisson solver via a tensor-product-based decomposition
- RRTMGP radiative transfer
- Boundary conditions: periodic in the horizontal, no-slip or free-slip in the
vertical
- Distributed checkpointing using Orbax
- Simulations are performed in FP32 precision
- Can run on TPU and GPU

## Benchmarking
Using Google TPUv6e (Trillium), benchmark performance results with $256^3$ grid
points per TPU core are as follows:

| # of TPU cores  | Wall time per timestep (ms) |
| --------------- | --------------------------- |
| 1               | 120                         |
| 2               | 124                         |
| 4               | 144                         |
| 8               | 178                         |
| 64              | 570                         |

In this benchmark, a single timestep comprises the 3 stages of the RK3
timestepper. Each stage consists of the equations for all prognostic variables,
the pressure Poisson solver, and the equilibrium thermodynamics nonlinear solve.
 RRTMGP is not included within the benchmark.

(Note that JAX's automatic parallelization is used, and no specific effort has
yet been made to optimize the performance at a large number of cores.)
