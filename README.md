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

- Anelastic momentum equations: velocities $u$, $v$, $w$
- Continuity equation: determines the pressure $p$ through a Poisson equation
- Linearized liquid-ice potential temperature $\theta_{li}$ (energy-like
variable)
- Total specific humidity $q_t$
- Mass fractions for 2 precipitation species, rain $q_r$ and snow $q_s$

## Features of Jatmos

- Staggered grid
- Conservative formulation
- Boundary conditions: periodic boundary in the horizontal, no-slip or free-slip
in the vertical
- Equilibrium thermodynamics for water phases
- One-moment microphysics
- Poisson solver via a tensor-product-based decomposition
- RRTMGP radiative transfer
- RK3 timestepper and adaptive timestepping based on a CFL condition
- Simulations are performed in FP32 precision

## Benchmarking
Using Google TPUv6e (Trillium), benchmark performance results with 256^3 grid
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
yet been made to optimize the communication)
