# `amangkurat`: Idealized (1+1)D nonlinear Klein-Gordon solver

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/badge/pypi-v0.0.1-orange.svg)](https://pypi.org/)
[![DOI](https://zenodo.org/badge/1081397514.svg)](https://doi.org/10.5281/zenodo.17624665)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NumPy](https://img.shields.io/badge/NumPy-%23013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?logo=scipy&logoColor=white)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?logo=Matplotlib&logoColor=black)](https://matplotlib.org/)
[![Numba](https://img.shields.io/badge/Numba-00A3E0?logo=numba&logoColor=white)](https://numba.pydata.org/)
[![NetCDF4](https://img.shields.io/badge/NetCDF4-blue)](https://unidata.github.io/netcdf4-python/)
[![tqdm](https://img.shields.io/badge/tqdm-FFC107?logo=tqdm&logoColor=black)](https://tqdm.github.io/)

## Overview

Numerical solver for the $(1+1)$-dimensional nonlinear Klein-Gordon equation using symplectic pseudo-spectral methods. Features adaptive timestepping, Numba acceleration, and scientific data output.

**Key Features:**
- Pseudo-spectral Fourier discretization (exponential convergence)
- Størmer-Verlet symplectic integration (structure-preserving)
- Adaptive CFL-based timestepping
- Parallel execution with Numba
- NetCDF4 data output + 3D animations

---

## Mathematical Foundation

### Klein-Gordon Equation

$$(1+1)$-dimensional relativistic field equation in natural units ($c = \hbar = 1$):

$$\frac{\partial^2 \phi}{\partial t^2} - \frac{\partial^2 \phi}{\partial x^2} + V'(\phi) = 0$$

### Supported Potentials

| Potential | $V(\phi)$ | $V'(\phi)$ | Physics |
|-----------|-----------|------------|---------|
| **Linear** | $\frac{1}{2}m^2\phi^2$ | $m^2\phi$ | Free massive scalar field |
| **$\phi^4$** | $\frac{\lambda}{4}(\phi^2 - v^2)^2$ | $\lambda\phi(\phi^2 - v^2)$ | Topological kinks, domain walls |
| **Sine-Gordon** | $1 - \cos(\phi)$ | $\sin(\phi)$ | Breathers, Josephson junctions |

### Numerical Method

**Spatial:** Pseudo-spectral Fourier with wavenumbers $k_n = 2\pi n/L$

**Temporal:** Størmer-Verlet (symplectic leapfrog)
$$\phi^{n+1} = 2\phi^n - \phi^{n-1} + \Delta t^2 \left[\nabla^2\phi^n - V'(\phi^n)\right]$$

**Stability:** CFL condition $\Delta t \leq 0.5/k_{\max}$ enforced via adaptive timestepping

---

## Installation

### From Source
```bash
git clone https://github.com/sandyherho/amangkurat.git
cd amangkurat
pip install -e .
```

### From PyPI
```bash
pip install amangkurat
```
**Dependencies:** numpy, scipy, matplotlib, netCDF4, tqdm, numba (optional, recommended for 10-100× speedup)

---

## Quick Start

### Command Line

```bash
amangkurat case1           # Linear wave
amangkurat case2           # Kink soliton
amangkurat case3           # Breather
amangkurat case4           # Kink-antikink collision

# Options
amangkurat case1 --cores 8 --output-dir results/ --quiet
amangkurat --all           # Run all cases
```

### Python API

```python
from amangkurat import KGSolver
from amangkurat.core.initial_conditions import KinkIC

# Initialize solver
solver = KGSolver(nx=512, x_min=-30.0, x_max=30.0, adaptive_dt=True, n_cores=4)

# Create initial condition
ic = KinkIC(vacuum=1.0, position=0.0, velocity=0.0)
phi0, phi_dot0 = ic(solver.x)

# Solve
result = solver.solve(
    phi0=phi0, phi_dot0=phi_dot0,
    dt=0.005, t_final=50.0,
    potential='phi4', lambda_=1.0, vacuum=1.0,
    n_snapshots=200
)

# Access results
x = result['x']      # Spatial grid
t = result['t']      # Time snapshots  
phi = result['phi']  # Field evolution (nt × nx)
```

---

## Test Cases

| Case | Potential | Initial Condition | Description |
|------|-----------|-------------------|-------------|
| 1 | Linear | Gaussian pulse | Dispersive wave propagation |
| 2 | $\phi^4$ | Static kink | Topological soliton ($\phi = v\tanh(x/\xi)$) |
| 3 | Sine-Gordon | Breather | Time-periodic localized oscillation |
| 4 | $\phi^4$ | Kink-antikink pair | Collision dynamics ($v = 0.3c$) |

---

## Configuration

Simple key-value format (`configs/case*.txt`):

```ini
scenario_name = Kink Soliton
nx = 1024
x_min = -50.0
x_max = 50.0
t_final = 50.0
dt = 0.005
adaptive_dt = true

potential = phi4
lambda = 1.0
vacuum = 1.0

ic_type = kink
position = 0.0
velocity = 0.0

n_cores = 0
save_netcdf = true
save_animation = true
n_frames = 200
fps = 30
colormap = inferno
```

---

## Output Files

**NetCDF4 data** (`outputs/*.nc`): 
- Variables: `x(nx)`, `t(nt)`, `phi(nt, nx)`
- Metadata: grid parameters, potential type, timestep info

**Animated GIFs** (`outputs/*.gif`):
- 3D visualization of $\phi(x,t)$ evolution
- Dynamic camera rotation, color-coded amplitude

**Log files** (`logs/*.log`):
- Detailed diagnostics, parameter listings, timing

---

## Physical Units

Code uses dimensionless units. Map to physical systems via scale factors:

### $\phi^4$ Kinks

**Natural scales:** Length $\xi = \sqrt{2}/v$, Time $T = \xi/c$, Energy $E_0 = \frac{2\sqrt{2}}{3}\lambda v^3$

| Context | Length Scale | Time Scale | Application |
|---------|-------------|------------|-------------|
| Particle physics | $\xi \sim 10^{-18}$ m | $T \sim 10^{-26}$ s | Higgs field (theoretical) |
| Condensed matter | $\xi \sim 10$ nm | $T \sim 1$ ps | Ferromagnetic domain walls |
| Josephson junctions | $\lambda_J \sim 1$ μm | $T \sim 10$ ps | Superconducting phase |

**Conversion:**
```python
units = result['units']
x_physical = result['x'] * length_scale_physical / units.length_scale
```

---

## Directory Structure

```
amangkurat/
├── configs/              # Test configurations
├── src/amangkurat/
│   ├── core/            # Solver + initial conditions
│   ├── io/              # NetCDF + config parser
│   ├── visualization/   # 3D animations
│   └── utils/           # Logger + timer
├── outputs/             # Results (*.nc, *.gif)
├── logs/                # Simulation logs
└── pyproject.toml       # Build config
```

---

## Limitations

- **1D spatial domain only** ($(1+1)$D spacetime)
- **Periodic boundary conditions** (not suitable for scattering)
- **Uniform grid** (no adaptive mesh refinement)
- **CFL-limited timestep** (explicit integration)
- **Requires smooth solutions** (spectral methods)

---

## Citation

```bibtex
@software{herho2025_amangkurat,
  author = {Herho, Sandy H. S.},
  title = {amangkurat: Idealized Nonlinear Klein-Gordon Solver},
  year = {2025},
  version = {0.0.1},
  url = {https://github.com/sandyherho/amangkurat}
}
```
