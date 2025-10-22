"""Klein-Gordon solver with symplectic integration."""

import numpy as np
from typing import Dict, Any, Optional, Callable
from tqdm import tqdm
import warnings
import os

try:
    from numba import njit, prange, set_num_threads
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])
    prange = range
    def set_num_threads(n):
        pass

warnings.filterwarnings('ignore')


@njit(parallel=True, cache=True)
def compute_force_parallel(phi: np.ndarray, laplacian: np.ndarray,
                          potential_deriv: np.ndarray) -> np.ndarray:
    """Compute acceleration with Numba parallelization."""
    result = np.empty_like(phi)
    for i in prange(len(phi)):
        result[i] = laplacian[i] - potential_deriv[i]
    return result


class KGSolver:
    """
    Klein-Gordon equation solver with symplectic integration.
    
    Solves: ∂²φ/∂t² - ∇²φ + V'(φ) = 0
    
    Features:
    - Spectral spatial derivatives
    - Störmer-Verlet time integration
    - Energy conservation
    - Numba parallelization
    """
    
    def __init__(self, nx: int = 512, x_min: float = -30.0, x_max: float = 30.0,
                 verbose: bool = True, logger: Optional[Any] = None,
                 n_cores: Optional[int] = None):
        self.nx = nx
        self.x_min = x_min
        self.x_max = x_max
        self.verbose = verbose
        self.logger = logger
        
        self.x = np.linspace(x_min, x_max, nx)
        self.dx = (x_max - x_min) / (nx - 1)
        
        self.k = 2.0 * np.pi * np.fft.fftfreq(nx, d=self.dx)
        
        if n_cores is None:
            n_cores = os.cpu_count()
        self.n_cores = n_cores
        
        if NUMBA_AVAILABLE:
            set_num_threads(self.n_cores)
        
        if verbose:
            print(f"  Grid: {nx} points, dx = {self.dx:.6f}")
            print(f"  Domain: [{x_min:.1f}, {x_max:.1f}]")
            print(f"  CPU cores: {self.n_cores}")
            print(f"  Numba: {'ENABLED' if NUMBA_AVAILABLE else 'DISABLED'}")
    
    def laplacian(self, phi: np.ndarray) -> np.ndarray:
        """Compute Laplacian using spectral method."""
        phi_hat = np.fft.fft(phi)
        lap_hat = -(self.k**2) * phi_hat
        return np.real(np.fft.ifft(lap_hat))
    
    def gradient(self, phi: np.ndarray) -> np.ndarray:
        """Compute gradient using spectral method."""
        phi_hat = np.fft.fft(phi)
        grad_hat = 1j * self.k * phi_hat
        return np.real(np.fft.ifft(grad_hat))
    
    def get_potential(self, pot_type: str, **params) -> Callable:
        """Get potential derivative function."""
        if pot_type == 'linear':
            mass = params.get('mass', 1.0)
            return lambda phi: mass**2 * phi
        
        elif pot_type == 'phi4':
            lam = params.get('lambda', 1.0)
            v = params.get('vacuum', 1.0)
            return lambda phi: lam * phi * (phi**2 - v**2)
        
        elif pot_type == 'sine_gordon':
            return lambda phi: np.sin(phi)
        
        else:
            raise ValueError(f"Unknown potential: {pot_type}")
    
    def stormer_verlet_step(self, phi: np.ndarray, phi_old: np.ndarray,
                           dt: float, potential_deriv: Callable) -> np.ndarray:
        """Single Störmer-Verlet time step."""
        lap = self.laplacian(phi)
        V_prime = potential_deriv(phi)
        
        if NUMBA_AVAILABLE:
            force = compute_force_parallel(phi, lap, V_prime)
        else:
            force = lap - V_prime
        
        phi_new = 2*phi - phi_old + dt**2 * force
        
        return phi_new
    
    def compute_energy(self, phi: np.ndarray, phi_dot: np.ndarray,
                      potential: Callable) -> float:
        """Compute total energy."""
        grad_phi = self.gradient(phi)
        
        kinetic = 0.5 * np.trapz(phi_dot**2, self.x)
        gradient_energy = 0.5 * np.trapz(grad_phi**2, self.x)
        
        if potential.__name__ == '<lambda>':
            V_vals = np.array([potential(p) for p in phi])
        else:
            V_vals = potential(phi)
        
        pot_energy = np.trapz(self._integrate_potential(phi, potential), self.x)
        
        return kinetic + gradient_energy + pot_energy
    
    def _integrate_potential(self, phi: np.ndarray, V_prime: Callable) -> np.ndarray:
        """Integrate V'(φ) to get V(φ) approximately."""
        if hasattr(V_prime, '__name__'):
            name = V_prime.__name__
            if 'linear' in str(name):
                return 0.5 * phi**2
            elif 'phi4' in str(name):
                return 0.25 * (phi**2 - 1)**2
            elif 'sine' in str(name):
                return 1 - np.cos(phi)
        return np.zeros_like(phi)
    
    def solve(self, phi0: np.ndarray, phi_dot0: np.ndarray,
              dt: float, t_final: float, potential: str = 'phi4',
              n_snapshots: int = 200, **pot_params) -> Dict[str, Any]:
        """
        Solve Klein-Gordon equation.
        
        Args:
            phi0: Initial field
            phi_dot0: Initial velocity
            dt: Time step
            t_final: Final time
            potential: Potential type
            n_snapshots: Output snapshots
            **pot_params: Potential parameters
        """
        if self.verbose:
            print(f"\n  Solving Klein-Gordon equation...")
            print(f"    Potential: {potential}")
            print(f"    dt = {dt}, t_final = {t_final}")
            print(f"    Method: Störmer-Verlet (symplectic)")
        
        V_prime = self.get_potential(potential, **pot_params)
        
        n_steps = int(t_final / dt)
        snapshot_every = max(1, n_steps // n_snapshots)
        
        phi = phi0.copy()
        phi_old = phi0 - dt * phi_dot0
        
        t_out = []
        phi_hist = []
        energy_hist = []
        
        if self.verbose:
            pbar = tqdm(range(n_steps), desc="  Integrating", unit="steps")
        else:
            pbar = range(n_steps)
        
        for step in pbar:
            t = step * dt
            
            if step % snapshot_every == 0:
                phi_dot = (phi - phi_old) / dt
                energy = self.compute_energy(phi, phi_dot, V_prime)
                
                t_out.append(t)
                phi_hist.append(phi.copy())
                energy_hist.append(energy)
            
            phi_new = self.stormer_verlet_step(phi, phi_old, dt, V_prime)
            
            phi_old = phi
            phi = phi_new
        
        if self.verbose and hasattr(pbar, 'close'):
            pbar.close()
        
        t_out = np.array(t_out)
        phi_hist = np.array(phi_hist)
        energy_hist = np.array(energy_hist)
        
        energy_error = np.abs((energy_hist - energy_hist[0]) / energy_hist[0]).max()
        
        if self.verbose:
            print(f"  ✓ Solution computed ({n_steps} steps)")
            print(f"    Energy error: {energy_error:.2e}")
        
        return {
            'x': self.x,
            't': t_out,
            'phi': phi_hist,
            'energy': energy_hist,
            'energy_error': energy_error,
            'params': {
                'nx': self.nx,
                'dx': self.dx,
                'dt': dt,
                'n_steps': n_steps,
                'potential': potential,
                'n_cores': self.n_cores,
                'numba_enabled': NUMBA_AVAILABLE,
                **pot_params
            }
        }
