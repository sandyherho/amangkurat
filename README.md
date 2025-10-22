# `amangkurat`: Relativistic Nonlinear Klein-Gordon Solver

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Numba](https://img.shields.io/badge/accelerated-numba-orange.svg)](https://numba.pydata.org/)

## Overview

`amangkurat` is a high-performance Python solver for the nonlinear Klein-Gordon equation—a fundamental relativistic field equation describing scalar field dynamics with self-interactions. The solver implements symplectic pseudo-spectral methods with Numba JIT acceleration, achieving exceptional energy conservation and parallel performance.

**Key Features:**
- Spectral accuracy in space (exponential convergence)
- Symplectic time integration (Störmer-Verlet)
- Energy-conserving dynamics
- 10-100× speedup with Numba parallelization
- Stunning 3D animated visualizations
- NetCDF4 scientific data output

## Physics

The nonlinear Klein-Gordon equation:

$$\frac{\partial^2 \phi}{\partial t^2} - \nabla^2 \phi + V'(\phi) = 0$$

**Common potentials:**
- **φ⁴ theory**: $V(\phi) = \lambda(\phi^2 - v^2)^2/4$ (kink solitons)
- **Sine-Gordon**: $V(\phi) = 1 - \cos(\phi)$ (breathers)
- **Linear**: $V(\phi) = m^2\phi^2/2$ (free massive field)

**Conservation Laws:**
1. **Energy**: $E = \int[\frac{1}{2}(\partial_t\phi)^2 + \frac{1}{2}(\nabla\phi)^2 + V(\phi)]dx$
2. **Momentum**: $P = \int \partial_t\phi \cdot \nabla\phi \, dx$

## Numerical Methods

### Spatial Discretization
Pseudo-spectral Fourier method with $O(e^{-cN})$ convergence for smooth solutions.

### Temporal Integration
**Störmer-Verlet (symplectic leapfrog)**:
```
φ^(n+1) = 2φ^n - φ^(n-1) + Δt²[∇²φ^n - V'(φ^n)]
```
- Symplectic: Preserves energy structure
- 2nd-order accurate
- Time-reversible
- Excellent long-time stability

### Parallelization
Numba JIT with `prange` for multi-core acceleration.

## Installation

```bash
# From source
git clone https://github.com/sandyherho/amangkurat-solver.git
cd amangkurat-solver
pip install -e .
```

## Quick Start

**Command line:**
```bash
amangkurat case1           # Linear wave
amangkurat case2           # Kink soliton  
amangkurat case3           # Breather
amangkurat case4 --cores 8 # Kink-antikink collision
```

**Python API:**
```python
from amangkurat import KGSolver

solver = KGSolver(nx=512, x_min=-30, x_max=30)
phi0 = solver.kink_initial_condition(amplitude=1.0)
result = solver.solve(phi0, dt=0.01, t_final=50.0, potential='phi4')
```

## Test Cases

| Case | Physics | Potential | Expected Behavior |
|------|---------|-----------|-------------------|
| 1 | Linear wave | $V = m^2\phi^2/2$ | Harmonic oscillation |
| 2 | Kink soliton | φ⁴ | Static topological defect |
| 3 | Breather | Sine-Gordon | Localized oscillation |
| 4 | Collision | φ⁴ | Complex interaction dynamics |

## Directory Structure

```
amangkurat-solver/
├── configs/              # Test case configurations
├── src/amangkurat/       # Main package
│   ├── core/            # Solver + initial conditions
│   ├── io/              # Config + NetCDF handlers
│   ├── visualization/   # 3D animation
│   └── utils/           # Logger + timer
├── outputs/             # Generated results
├── logs/                # Simulation logs
└── pyproject.toml       # Build config
```

## Citation

```bibtex
@software{herho2025_amangkurat,
  author = {Herho, Sandy H. S.},
  title = {amangkurat: Nonlinear Klein-Gordon Solver},
  year = {2025},
  url = {https://github.com/sandyherho/amangkurat-solver}
}
```

## Author

Sandy H. S. Herho

## License

MIT License - see LICENSE file
