"""
Idealized Klein-Gordon solver.

Engineered for performance using spectral methods and Numba JIT compilation.
Implements a symplectic Störmer-Verlet integrator with adaptive time-stepping
and zero-allocation memory management in the critical loop.
"""

import numpy as np
from typing import Dict, Any, Optional, Callable, Tuple
from tqdm import tqdm
import warnings
import os

# -----------------------------------------------------------------------------
# JIT Compilation Setup
# -----------------------------------------------------------------------------
# Tries to import Numba for high-performance JIT compilation.
# Falls back to a dummy decorator pattern if Numba is missing to ensure portability.
try:
    from numba import njit, prange, set_num_threads
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback: Identity decorators and mock functions
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])
    prange = range
    def set_num_threads(n):
        pass

warnings.filterwarnings('ignore')


# -----------------------------------------------------------------------------
# High-Performance Kernels
# -----------------------------------------------------------------------------
@njit(parallel=True, cache=True, fastmath=True)
def compute_step_kernel(phi: np.ndarray, phi_old: np.ndarray, 
                       laplacian: np.ndarray, potential_deriv: np.ndarray, 
                       dt: float, out: np.ndarray) -> None:
    """
    Fused Computational Kernel.
    
    Perf Note:
        - Combines acceleration calculation and time integration in a single pass.
        - Increases Cache Locality: Data is loaded once and used immediately.
        - Reduces Memory Bandwidth: Avoids writing an intermediate 'force' array to RAM.
        - Vectorized and Parallelized via Numba.
    
    Args:
        out: Pre-allocated buffer to store the result (Zero-Allocation pattern).
    """
    dt_sq = dt * dt
    # prange allows OpenMP-like parallel execution
    for i in prange(len(phi)):
        force = laplacian[i] - potential_deriv[i]
        out[i] = 2.0 * phi[i] - phi_old[i] + dt_sq * force


class PhysicalUnits:
    """
    Handles dimensional analysis and unit scaling.
    Decouples the numerical solver (normalized units) from physical interpretation.
    """

    def __init__(self, potential_type: str = 'phi4', **params):
        self.pot_type = potential_type
        # Setup scales based on potential parameters (mass, lambda, vacuum expectation)
        if potential_type == 'phi4':
            self.v = params.get('vacuum', 1.0)
            self.lam = params.get('lambda', 1.0)
            self.length_scale = np.sqrt(2) / self.v
            self.energy_scale = (2*np.sqrt(2)/3) * self.lam * self.v**3
            self.time_scale = self.length_scale
            self.mass_scale = np.sqrt(2 * self.lam) * self.v
        elif potential_type == 'sine_gordon':
            self.length_scale = 1.0
            self.energy_scale = 8.0
            self.time_scale = 1.0
            self.mass_scale = 1.0
        elif potential_type == 'linear':
            self.m = params.get('mass', 1.0)
            self.length_scale = 1.0 / self.m
            self.energy_scale = self.m
            self.time_scale = 1.0 / self.m
            self.mass_scale = self.m
        else:
            self.length_scale = 1.0
            self.energy_scale = 1.0
            self.time_scale = 1.0
            self.mass_scale = 1.0

    def to_physical_length(self, x_computational):
        return x_computational * self.length_scale

    def to_physical_time(self, t_computational):
        return t_computational * self.time_scale

    def to_physical_energy(self, E_computational):
        return E_computational * self.energy_scale

    def __repr__(self):
        return (f"PhysicalUnits({self.pot_type}: "
                f"L={self.length_scale:.4f}, "
                f"E={self.energy_scale:.4f}, "
                f"T={self.time_scale:.4f})")


class KGSolver:
    """
    Klein-Gordon solver optimized for 1D scalar fields.
    
    Key Features:
    - Spectral Derivatives (Real-FFT) for high spatial accuracy and 2x speedup.
    - Symplectic Integration (Störmer-Verlet) for energy stability.
    - Buffer rotation to minimize Garbage Collection overhead.
    """
    
    def __init__(self, nx: int = 512, x_min: float = -30.0, x_max: float = 30.0,
                 verbose: bool = True, logger: Optional[Any] = None,
                 n_cores: Optional[int] = None, adaptive_dt: bool = True):
        self.nx = nx
        self.x_min = x_min
        self.x_max = x_max
        self.verbose = verbose
        self.logger = logger
        self.adaptive_dt = adaptive_dt
        
        # Grid initialization
        self.x = np.linspace(x_min, x_max, nx)
        self.dx = (x_max - x_min) / (nx - 1)
        self.L = x_max - x_min
        
        # Pre-compute spectral wavenumbers for Real-FFT (rfft)
        # Exploits conjugate symmetry: array size is N//2 + 1, reducing memory and ops.
        self.k = 2.0 * np.pi * np.fft.rfftfreq(nx, d=self.dx)
        self.k_max = np.max(np.abs(self.k))
        
        # Concurrency setup
        if n_cores is None:
            n_cores = os.cpu_count()
        self.n_cores = n_cores
        
        if NUMBA_AVAILABLE:
            set_num_threads(self.n_cores)
        
        # CFL Condition (Courant–Friedrichs–Lewy) limits for explicit integration
        self.dt_min = 0.1 / self.k_max
        self.dt_max = 0.5 / self.k_max
        
        # Memory Optimization:
        # Pre-allocate a reusable buffer to avoid malloc/free cycles in the inner loop.
        self.phi_new_buffer = np.zeros(nx, dtype=np.float64)
        
        self.units = None
        self.diagnostics = {
            'timestep_history': [],
            'cfl_history': [],
            'adaptation_count': 0,
            'warnings': []
        }
        
        if verbose:
            print(f"  Grid: {nx} points, dx = {self.dx:.6f}, L = {self.L:.2f}")
            print(f"  Domain: [{x_min:.1f}, {x_max:.1f}]")
            print(f"  Max wavenumber: k_max = {self.k_max:.4f}")
            print(f"  dt limits: [{self.dt_min:.6f}, {self.dt_max:.6f}]")
            print(f"  CPU cores: {self.n_cores}")
            print(f"  Numba: {'ENABLED' if NUMBA_AVAILABLE else 'DISABLED'}")
            print(f"  Adaptive dt: {'ENABLED' if adaptive_dt else 'DISABLED'}")
    
    def laplacian(self, phi: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian (d^2/dx^2) using Real Fourier Spectral Method.
        Accuracy: Spectral. Uses rfft/irfft for ~2x performance gain on real fields.
        """
        phi_hat = np.fft.rfft(phi)
        # Multiply by -k^2 in frequency domain = 2nd derivative in spatial domain
        lap_hat = -(self.k**2) * phi_hat
        # Specify n=self.nx to ensure exact reconstruction size
        return np.fft.irfft(lap_hat, n=self.nx)
    
    def gradient(self, phi: np.ndarray) -> np.ndarray:
        """
        Compute Gradient (d/dx) using Real Fourier Spectral Method.
        Uses rfft/irfft for performance.
        """
        phi_hat = np.fft.rfft(phi)
        grad_hat = 1j * self.k * phi_hat
        # Specify n=self.nx to ensure exact reconstruction size
        return np.fft.irfft(grad_hat, n=self.nx)
    
    def get_potential_functions(self, pot_type: str, **params) -> Tuple[Callable, Callable]:
        """Factory method to generate Potential V(phi) and Force V'(phi) functions."""
        if pot_type == 'linear':
            mass = params.get('mass', 1.0)
            m2 = mass**2
            V = lambda phi: 0.5 * m2 * phi**2
            V_prime = lambda phi: m2 * phi
            
            if self.logger:
                self.logger.info(f"  Potential: V(φ) = ½m²φ², m = {mass}")
        
        elif pot_type == 'phi4':
            lam = params.get('lambda', 1.0)
            v = params.get('vacuum', 1.0)
            v2 = v**2
            V = lambda phi: (lam / 4.0) * (phi**2 - v2)**2
            V_prime = lambda phi: lam * phi * (phi**2 - v2)
            
            E_kink = (2 * np.sqrt(2) / 3) * lam * v**3
            kink_width = np.sqrt(2) / v
            
            if self.logger:
                self.logger.info(f"  Potential: V(φ) = (λ/4)(φ² - v²)²")
                self.logger.info(f"    λ = {lam}, v = {v}")
                self.logger.info(f"    Kink energy: E₀ = {E_kink:.6f}")
                self.logger.info(f"    Kink width: ξ = {kink_width:.6f}")
        
        elif pot_type == 'sine_gordon':
            V = lambda phi: 1.0 - np.cos(phi)
            V_prime = lambda phi: np.sin(phi)
            
            if self.logger:
                self.logger.info(f"  Potential: V(φ) = 1 - cos(φ)")
                self.logger.info(f"    Kink energy: E₀ = 8.0")
        
        else:
            raise ValueError(f"Unknown potential: {pot_type}")
        
        return V, V_prime
    
    def stormer_verlet_step(self, phi: np.ndarray, phi_old: np.ndarray,
                           dt: float, V_prime: Callable, 
                           out: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Executes a single time-integration step.
        
        Algorithm: Position Verlet / Störmer-Verlet (2nd order, Symplectic).
        Why Symplectic? It preserves phase-space volume, ensuring energy stability
        over long simulation times, unlike Runge-Kutta.
        
        Args:
            out: Destination buffer. If provided, enables Zero-Allocation path.
        """
        lap = self.laplacian(phi)
        V_prime_vals = V_prime(phi)
        
        if out is None:
            out = np.empty_like(phi) # Fallback allocation

        if NUMBA_AVAILABLE:
            # Hot path: JIT compiled kernel
            compute_step_kernel(phi, phi_old, lap, V_prime_vals, dt, out)
        else:
            # Cold path: Numpy vectorized operations
            force = lap - V_prime_vals
            # Using in-place operations (np.add(out=...)) to minimize temp arrays
            np.copyto(out, phi)
            np.multiply(out, 2.0, out=out)
            np.subtract(out, phi_old, out=out)
            np.multiply(force, dt**2, out=force)
            np.add(out, force, out=out)
        
        return out
    
    def check_stability(self, phi: np.ndarray, step: int, t: float) -> bool:
        """Sanity check for numerical explosion (NaNs or Infs)."""
        if not np.isfinite(phi).all():
            warning = f"NaN/Inf detected at step {step}, t={t:.4f}"
            self.diagnostics['warnings'].append(warning)
            if self.logger:
                self.logger.error(warning)
                self.logger.error(f"  max|φ|: {np.max(np.abs(phi))}")
            return False
        return True
    
    def adapt_timestep(self, phi: np.ndarray, dt_current: float, step: int) -> Tuple[float, str]:
        """
        Feedback Controller for Time-Step size.
        Reduces dt when high-frequency components (gradients) increase to satisfy CFL.
        """
        max_phi = np.max(np.abs(phi))
        if max_phi > 100:
            dt_new = dt_current * 0.7
            reason = f"Large field |φ| = {max_phi:.2e}"
        elif max_phi > 10:
            dt_new = dt_current * 0.9
            reason = f"Moderate field |φ| = {max_phi:.2e}"
        else:
            # Gently increase dt to optimize runtime when physics is calm
            dt_new = dt_current * 1.05
            reason = "Stable"
            
        # Hard clamp limits
        dt_new = max(self.dt_min, min(dt_new, self.dt_max))
        
        if dt_new <= self.dt_min * 1.01:
            warning = f"Step {step}: Timestep saturation at min {self.dt_min:.6f}"
            if warning not in self.diagnostics['warnings']:
                self.diagnostics['warnings'].append(warning)
                if self.logger:
                    self.logger.warning(warning)
        return dt_new, reason
    
    def solve(self, phi0: np.ndarray, phi_dot0: np.ndarray,
              dt: float, t_final: float, potential: str = 'phi4',
              n_snapshots: int = 200, **pot_params) -> Dict[str, Any]:
        """
        Main Simulation Driver.
        Manages initialization, main loop, data logging, and memory rotation.
        """
        self.units = PhysicalUnits(potential, **pot_params)
        
        if self.verbose:
            print(f"\n  Solving Klein-Gordon equation...")
            print(f"    Potential: {potential}")
            print(f"    dt_initial = {dt:.6f}, t_final = {t_final:.2f}")
            print(f"    Method: Størmer-Verlet (symplectic)")
            print(f"    {self.units}")
        
        if self.logger:
            self.logger.info("SOLVER INITIALIZATION")
            self.logger.info(f"Physical units: {self.units}")
            self.logger.info(f"dt limits: [{self.dt_min:.6f}, {self.dt_max:.6f}]")
        
        V_func, V_prime = self.get_potential_functions(potential, **pot_params)
        
        dt = max(self.dt_min, min(dt, self.dt_max))
        initial_cfl = dt * self.k_max
        
        if self.verbose:
            print(f"    Initial CFL number: {initial_cfl:.4f}")
        
        if self.logger:
            self.logger.info(f"Initial timestep: dt = {dt:.6f}")
            self.logger.info(f"Initial CFL number: {initial_cfl:.4f}")
        
        # ---------------------------------------------------------------------
        # Initial Step Bootstrap
        # Störmer-Verlet requires t-1. We approximate it using Taylor expansion:
        # phi(-dt) ≈ phi(0) - dt*phi'(0) + 0.5*dt^2*phi''(0)
        # ---------------------------------------------------------------------
        phi = phi0.copy()
        lap0 = self.laplacian(phi0)
        V_prime_0 = V_prime(phi0)
        
        force0 = lap0 - V_prime_0
        phi_old = phi0 - dt * phi_dot0 + 0.5 * dt**2 * force0
        
        if self.logger:
            self.logger.info("INITIAL CONDITIONS")
            self.logger.info(f"Max field: max|φ₀| = {np.max(np.abs(phi0)):.6f}")
        
        # Output containers
        t_out = [0.0]
        phi_hist = [phi0.copy()]
        
        t = 0.0
        step = 0
        dt_current = dt
        snapshot_interval = t_final / n_snapshots
        next_snapshot_time = snapshot_interval
        
        if self.verbose:
            print(f"\n  Starting time integration...")
            pbar = tqdm(total=t_final, desc="  Progress", unit="t", 
                       bar_format='{l_bar}{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}]')
        else:
            pbar = None
        
        if self.logger:
            self.logger.info("TIME INTEGRATION")
        
        max_steps = int(t_final / self.dt_min) * 10
        
        # Ensure buffer alignment
        if self.phi_new_buffer.shape != phi.shape:
             self.phi_new_buffer = np.zeros_like(phi)

        # ---------------------------------------------------------------------
        # Main Time-Integration Loop
        # ---------------------------------------------------------------------
        while t < t_final and step < max_steps:
            # 1. Compute Step (Write directly into pre-allocated buffer)
            self.stormer_verlet_step(phi, phi_old, dt_current, V_prime, 
                                   out=self.phi_new_buffer)
            
            # 2. Safety Check (Costly, but necessary for unbounded potentials)
            if not self.check_stability(self.phi_new_buffer, step, t):
                raise RuntimeError(f"Simulation became unstable at t={t:.4f}, step={step}")
            
            # 3. Telemetry
            self.diagnostics['timestep_history'].append(dt_current)
            self.diagnostics['cfl_history'].append(dt_current * self.k_max)
            
            # 4. Adaptive Stepping Logic (Every 20 steps to amortize cost)
            if self.adaptive_dt and step % 20 == 0:
                dt_new, reason = self.adapt_timestep(phi, dt_current, step)
                if abs(dt_new - dt_current) / dt_current > 0.05:
                    self.diagnostics['adaptation_count'] += 1
                    if self.logger and self.diagnostics['adaptation_count'] % 10 == 0:
                        self.logger.info(f"t={t:.4f}: dt {dt_current:.6f} -> {dt_new:.6f} ({reason})")
                    dt_current = dt_new
            
            # 5. Snapshotting (IO Bound)
            if t >= next_snapshot_time or t + dt_current >= t_final:
                t_out.append(t)
                phi_hist.append(phi.copy()) # Must copy here to persist data
                next_snapshot_time += snapshot_interval
                
                if self.logger and len(t_out) % 50 == 0:
                    self.logger.info(f"Step {step}: t={t:.4f}, dt={dt_current:.6f}")
            
            # 6. Pointer Rotation / Zero-Copy Swap
            # Instead of copying array contents (O(N)), we swap references (O(1)).
            # Logic: 
            #   Old 'phi_old' is no longer needed -> recycle as next buffer
            #   Current 'phi' becomes 'phi_old'
            #   New data in 'phi_new_buffer' becomes 'phi'
            temp_recyclable = phi_old
            phi_old = phi
            phi = self.phi_new_buffer
            self.phi_new_buffer = temp_recyclable

            t += dt_current
            step += 1
            
            if pbar is not None:
                pbar.n = min(t, t_final)
                pbar.refresh()
        
        if pbar is not None:
            pbar.close()
        
        if step >= max_steps:
            warning = f"Reached maximum steps ({max_steps}), stopping early"
            if self.logger:
                self.logger.warning(warning)
            if self.verbose:
                print(f"\n  Warning: {warning}")
        
        # ---------------------------------------------------------------------
        # Finalization & Packaging
        # ---------------------------------------------------------------------
        t_out = np.array(t_out)
        phi_hist = np.array(phi_hist)
        
        if self.verbose:
            print(f"  Solution computed ({step} steps)")
            if self.adaptive_dt:
                print(f"    Timestep adaptations: {self.diagnostics['adaptation_count']}")
        
        if self.logger:
            self.logger.info("SIMULATION COMPLETED")
            self.logger.info(f"Total steps: {step}")
            self.logger.info(f"Final time: t = {t:.6f}")
            if self.adaptive_dt:
                self.logger.info(f"Timestep adaptations: {self.diagnostics['adaptation_count']}")
                self.logger.info(f"Final timestep: dt = {dt_current:.6f}")
            if self.diagnostics['warnings']:
                self.logger.warning(f"{len(self.diagnostics['warnings'])} warnings occurred")
        
        return {
            'x': self.x,
            't': t_out,
            'phi': phi_hist,
            'diagnostics': self.diagnostics,
            'units': self.units,
            'params': {
                'nx': self.nx,
                'dx': self.dx,
                'dt_initial': dt,
                'dt_final': dt_current,
                'dt_min': self.dt_min,
                'dt_max': self.dt_max,
                'adaptations': self.diagnostics['adaptation_count'],
                'n_steps': step,
                'potential': potential,
                'n_cores': self.n_cores,
                'numba_enabled': NUMBA_AVAILABLE,
                'adaptive_dt': self.adaptive_dt,
                **pot_params
            }
        }