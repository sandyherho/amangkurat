"""Klein-Gordon solver"""

import numpy as np
from typing import Dict, Any, Optional, Callable, Tuple
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


class PhysicalUnits:
    """Physical units for Klein-Gordon equation.

    Natural units where c = ℏ = 1.

    For φ⁴ theory: V(φ) = (λ/4)(φ² - v²)²
        - Length scale: ξ = √2/v  (kink width)
        - Energy scale: E_kink = (2√2/3)λv³  (single kink energy)
        - Time scale: T = ξ  (light-crossing time)
        - Mass scale: m = √(2λ)v  (small oscillation frequency)

    For sine-Gordon: V(φ) = 1 - cos(φ)
        - Length scale: 1 (natural units)
        - Energy scale: 8 (kink energy)
        - Time scale: 1

    For linear: V(φ) = ½m²φ²
        - Length scale: 1/m (Compton wavelength)
        - Energy scale: m (mass)
        - Time scale: 1/m (Compton time)
    """

    def __init__(self, potential_type: str = 'phi4', **params):
        self.pot_type = potential_type

        if potential_type == 'phi4':
            self.v = params.get('vacuum', 1.0)
            self.lam = params.get('lambda', 1.0)
            # Natural scales from kink solution
            self.length_scale = np.sqrt(2) / self.v  # Kink width ξ
            self.energy_scale = (2*np.sqrt(2)/3) * self.lam * self.v**3  # E_kink
            self.time_scale = self.length_scale  # Light-crossing time
            self.mass_scale = np.sqrt(2 * self.lam) * self.v  # Small oscillation mass
        elif potential_type == 'sine_gordon':
            # Sine-Gordon in natural units
            self.length_scale = 1.0
            self.energy_scale = 8.0  # Kink energy
            self.time_scale = 1.0
            self.mass_scale = 1.0
        elif potential_type == 'linear':
            self.m = params.get('mass', 1.0)
            # Compton scales
            self.length_scale = 1.0 / self.m  # λ_C = ℏ/(mc)
            self.energy_scale = self.m  # mc²
            self.time_scale = 1.0 / self.m  # ℏ/(mc²)
            self.mass_scale = self.m
        else:
            self.length_scale = 1.0
            self.energy_scale = 1.0
            self.time_scale = 1.0
            self.mass_scale = 1.0

    def to_physical_length(self, x_computational):
        """Convert computational length to physical length."""
        return x_computational * self.length_scale

    def to_physical_time(self, t_computational):
        """Convert computational time to physical time."""
        return t_computational * self.time_scale

    def to_physical_energy(self, E_computational):
        """Convert computational energy to physical energy."""
        return E_computational * self.energy_scale

    def __repr__(self):
        return (f"PhysicalUnits({self.pot_type}: "
                f"L={self.length_scale:.4f}, "
                f"E={self.energy_scale:.4f}, "
                f"T={self.time_scale:.4f})")


class KGSolver:
    """Klein-Gordon solver with FIXED adaptive timestepping."""
    
    def __init__(self, nx: int = 512, x_min: float = -30.0, x_max: float = 30.0,
                 verbose: bool = True, logger: Optional[Any] = None,
                 n_cores: Optional[int] = None, adaptive_dt: bool = True,
                 energy_tol: float = 1e-4):
        self.nx = nx
        self.x_min = x_min
        self.x_max = x_max
        self.verbose = verbose
        self.logger = logger
        self.adaptive_dt = adaptive_dt
        self.energy_tol = energy_tol
        
        # Spatial grid
        self.x = np.linspace(x_min, x_max, nx)
        self.dx = (x_max - x_min) / (nx - 1)
        self.L = x_max - x_min
        
        # Spectral wavenumbers
        self.k = 2.0 * np.pi * np.fft.fftfreq(nx, d=self.dx)
        self.k_max = np.max(np.abs(self.k))
        
        # Parallelization
        if n_cores is None:
            n_cores = os.cpu_count()
        self.n_cores = n_cores
        
        if NUMBA_AVAILABLE:
            set_num_threads(self.n_cores)
        
        # ROBUST: Compute minimum and maximum allowed timesteps
        self.dt_min = 0.1 / self.k_max  # 5× safety factor on CFL
        self.dt_max = 0.5 / self.k_max  # CFL limit
        
        self.units = None
        self.diagnostics = {
            'energy_history': [],
            'momentum_history': [],
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
            if adaptive_dt:
                print(f"  Energy tolerance: {energy_tol:.2e}")
    
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
    
    def get_potential_functions(self, pot_type: str, **params) -> Tuple[Callable, Callable]:
        """Get potential and its derivative."""
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
    
    def compute_energy(self, phi: np.ndarray, phi_dot: np.ndarray,
                      V_func: Callable) -> float:
        """Compute total energy."""
        grad_phi = self.gradient(phi)

        kinetic = 0.5 * np.trapz(phi_dot**2, self.x)
        gradient_energy = 0.5 * np.trapz(grad_phi**2, self.x)
        V_vals = V_func(phi)
        pot_energy = np.trapz(V_vals, self.x)

        return kinetic + gradient_energy + pot_energy
    
    def compute_momentum(self, phi: np.ndarray, phi_dot: np.ndarray) -> float:
        """Compute total momentum."""
        grad_phi = self.gradient(phi)
        return np.trapz(phi_dot * grad_phi, self.x)
    
    def stormer_verlet_step(self, phi: np.ndarray, phi_old: np.ndarray,
                           dt: float, V_prime: Callable) -> np.ndarray:
        """Single Störmer-Verlet time step."""
        lap = self.laplacian(phi)
        V_prime_vals = V_prime(phi)
        
        if NUMBA_AVAILABLE:
            force = compute_force_parallel(phi, lap, V_prime_vals)
        else:
            force = lap - V_prime_vals
        
        phi_new = 2*phi - phi_old + dt**2 * force
        
        return phi_new
    
    def adapt_timestep(self, energy_current: float, energy_initial: float,
                      dt_current: float, step: int) -> Tuple[float, str]:
        """
        ROBUST adaptive timestep control.
        """
        rel_energy_error = abs((energy_current - energy_initial) / (energy_initial + 1e-10))
        
        # CONSERVATIVE: Less aggressive adaptation
        if rel_energy_error > 10 * self.energy_tol:
            dt_new = dt_current * 0.7  # Was 0.5
            reason = f"Large error {rel_energy_error:.2e}"
        elif rel_energy_error > 2 * self.energy_tol:
            dt_new = dt_current * 0.9  # Was 0.8
            reason = f"Moderate error {rel_energy_error:.2e}"
        elif rel_energy_error < self.energy_tol / 3:
            dt_new = dt_current * 1.05  # Was 1.1
            reason = f"Good conservation {rel_energy_error:.2e}"
        else:
            dt_new = dt_current
            reason = "OK"
        
        # CRITICAL: Enforce limits
        dt_new = max(self.dt_min, min(dt_new, self.dt_max))
        
        # Warn if hitting minimum
        if dt_new <= self.dt_min * 1.01:
            warning = f"Step {step}: Timestep at minimum {self.dt_min:.6f}"
            if warning not in self.diagnostics['warnings']:
                self.diagnostics['warnings'].append(warning)
                if self.logger:
                    self.logger.warning(warning)
        
        return dt_new, reason
    
    def solve(self, phi0: np.ndarray, phi_dot0: np.ndarray,
              dt: float, t_final: float, potential: str = 'phi4',
              n_snapshots: int = 200, **pot_params) -> Dict[str, Any]:
        """
        Solve Klein-Gordon equation with FIXED adaptive timestepping.
        
        KEY FIX: Velocity calculation now uses centered difference to ensure
        both position and velocity are evaluated at the same time level.
        """
        self.units = PhysicalUnits(potential, **pot_params)
        
        if self.verbose:
            print(f"\n  Solving Klein-Gordon equation...")
            print(f"    Potential: {potential}")
            print(f"    dt_initial = {dt:.6f}, t_final = {t_final:.2f}")
            print(f"    Method: Störmer-Verlet (symplectic)")
            print(f"    {self.units}")
        
        if self.logger:
            self.logger.info("=" * 60)
            self.logger.info("SOLVER INITIALIZATION")
            self.logger.info("=" * 60)
            self.logger.info(f"Physical units: {self.units}")
            self.logger.info(f"dt limits: [{self.dt_min:.6f}, {self.dt_max:.6f}]")
        
        # Get potential functions
        V_func, V_prime = self.get_potential_functions(potential, **pot_params)
        
        # ROBUST: Ensure initial timestep is safe
        dt = max(self.dt_min, min(dt, self.dt_max))
        initial_cfl = dt * self.k_max
        
        if self.verbose:
            print(f"    Initial CFL number: {initial_cfl:.4f}")
        
        if self.logger:
            self.logger.info(f"Initial timestep: dt = {dt:.6f} (enforced limits)")
            self.logger.info(f"Initial CFL number: {initial_cfl:.4f}")
        
        # ====================================================================
        # FIX #1: Second-order accurate initialization
        # ====================================================================
        phi = phi0.copy()
        
        # Compute initial force for second-order backward step
        lap0 = self.laplacian(phi0)
        V_prime_0 = V_prime(phi0)
        if NUMBA_AVAILABLE:
            force0 = compute_force_parallel(phi0, lap0, V_prime_0)
        else:
            force0 = lap0 - V_prime_0
        
        # Second-order accurate initialization (was first-order before)
        phi_old = phi0 - dt * phi_dot0 + 0.5 * dt**2 * force0
        
        # Initial quantities
        energy_initial = self.compute_energy(phi0, phi_dot0, V_func)
        momentum_initial = self.compute_momentum(phi0, phi_dot0)
        
        if self.verbose:
            print(f"    Initial energy: E₀ = {energy_initial:.8f}")
            print(f"    Initial momentum: P₀ = {momentum_initial:.8f}")
        
        if self.logger:
            self.logger.info("=" * 60)
            self.logger.info("INITIAL CONDITIONS")
            self.logger.info("=" * 60)
            self.logger.info(f"Energy: E₀ = {energy_initial:.12f} (computational)")
            E_phys = self.units.to_physical_energy(energy_initial)
            self.logger.info(f"        E₀ = {E_phys:.12f} (physical units)")
            self.logger.info(f"        E₀/E_kink = {energy_initial/self.units.energy_scale:.4f}")
            self.logger.info(f"Momentum: P₀ = {momentum_initial:.12f}")
            self.logger.info(f"Max field: max|φ₀| = {np.max(np.abs(phi0)):.6f}")
        
        # Storage
        t_out = [0.0]
        phi_hist = [phi0.copy()]
        energy_hist = [energy_initial]
        momentum_hist = [momentum_initial]
        
        # Integration
        t = 0.0
        step = 0
        dt_current = dt
        snapshot_interval = t_final / n_snapshots
        next_snapshot_time = snapshot_interval
        
        max_energy_error = 0.0
        max_momentum_error = 0.0
        
        if self.verbose:
            print(f"\n  Starting time integration...")
            pbar = tqdm(total=t_final, desc="  Progress", unit="t")
        else:
            pbar = None
        
        if self.logger:
            self.logger.info("=" * 60)
            self.logger.info("TIME INTEGRATION")
            self.logger.info("=" * 60)
        
        # ROBUST: Add step limit to prevent infinite loops
        max_steps = int(t_final / self.dt_min) * 10  # 10× safety margin
        
        # ====================================================================
        # FIX #2: Velocity computed with centered difference
        # ====================================================================
        while t < t_final and step < max_steps:
            # Störmer-Verlet step
            phi_new = self.stormer_verlet_step(phi, phi_old, dt_current, V_prime)
            
            # KEY FIX: Compute velocity at time n using CENTERED difference
            # This ensures phi and phi_dot are at the same time level!
            # phi_new is at time n+1, phi is at time n, phi_old is at time n-1
            # Velocity at time n: v_n = (phi_{n+1} - phi_{n-1}) / (2*dt)
            phi_dot = (phi_new - phi_old) / (2.0 * dt_current)
            
            # Diagnostics at time n (both phi and phi_dot at same time now!)
            energy = self.compute_energy(phi, phi_dot, V_func)
            momentum = self.compute_momentum(phi, phi_dot)
            
            energy_error = abs((energy - energy_initial) / (energy_initial + 1e-10))
            momentum_error = abs((momentum - momentum_initial) / (abs(momentum_initial) + 1e-10))
            
            max_energy_error = max(max_energy_error, energy_error)
            max_momentum_error = max(max_momentum_error, momentum_error)
            
            self.diagnostics['energy_history'].append(energy)
            self.diagnostics['momentum_history'].append(momentum)
            self.diagnostics['timestep_history'].append(dt_current)
            self.diagnostics['cfl_history'].append(dt_current * self.k_max)
            
            # ROBUST: Check for catastrophic failure
            if not np.isfinite(energy) or not np.isfinite(phi).all():
                error_msg = f"NaN/Inf detected at t={t:.4f}, step={step}"
                if self.logger:
                    self.logger.error(error_msg)
                    self.logger.error(f"  Energy: {energy}")
                    self.logger.error(f"  max|φ|: {np.max(np.abs(phi))}")
                    self.logger.error(f"  dt: {dt_current}")
                    self.logger.error(f"  Last 10 timesteps: {self.diagnostics['timestep_history'][-10:]}")
                    self.logger.error(f"  Last 10 energies: {self.diagnostics['energy_history'][-10:]}")
                raise RuntimeError(error_msg)
            
            # Adaptive timestepping - ONLY every 20 steps (not 10)
            if self.adaptive_dt and step % 20 == 0:
                dt_new, reason = self.adapt_timestep(energy, energy_initial, dt_current, step)
                
                if abs(dt_new - dt_current) / dt_current > 0.05:  # 5% change
                    self.diagnostics['adaptation_count'] += 1
                    if self.logger and self.diagnostics['adaptation_count'] % 10 == 0:
                        self.logger.info(f"t={t:.4f}: dt {dt_current:.6f} → {dt_new:.6f} ({reason})")
                    dt_current = dt_new
            
            # Save snapshot
            if t >= next_snapshot_time or t + dt_current >= t_final:
                t_out.append(t)
                phi_hist.append(phi.copy())
                energy_hist.append(energy)
                momentum_hist.append(momentum)
                next_snapshot_time += snapshot_interval
                
                if self.logger and len(t_out) % 50 == 0:
                    self.logger.info(f"Step {step}: t={t:.4f}, E={energy:.8f}, "
                                   f"ΔE/E₀={energy_error:.2e}, dt={dt_current:.6f}")
            
            # Update for next iteration (AFTER computing diagnostics!)
            phi_old = phi
            phi = phi_new
            t += dt_current
            step += 1
            
            # Update progress
            if pbar is not None:
                pbar.n = min(t, t_final)
                pbar.refresh()
        
        if pbar is not None:
            pbar.close()
        
        # Check if we hit step limit
        if step >= max_steps:
            warning = f"Reached maximum steps ({max_steps}), stopping early"
            if self.logger:
                self.logger.warning(warning)
            if self.verbose:
                print(f"\n  ⚠️  {warning}")
        
        # Convert to arrays
        t_out = np.array(t_out)
        phi_hist = np.array(phi_hist)
        energy_hist = np.array(energy_hist)
        momentum_hist = np.array(momentum_hist)
        
        # Final report
        if self.verbose:
            print(f"  ✓ Solution computed ({step} steps)")
            print(f"    Max energy error: ΔE/E₀ = {max_energy_error:.2e}")
            print(f"    Max momentum error: ΔP/P₀ = {max_momentum_error:.2e}")
            if self.adaptive_dt:
                print(f"    Timestep adaptations: {self.diagnostics['adaptation_count']}")
        
        if self.logger:
            self.logger.info("=" * 60)
            self.logger.info("SIMULATION COMPLETED")
            self.logger.info("=" * 60)
            self.logger.info(f"Total steps: {step}")
            self.logger.info(f"Final time: t = {t:.6f}")
            self.logger.info(f"Energy conservation: ΔE/E₀ = {max_energy_error:.3e}")
            self.logger.info(f"Momentum conservation: ΔP/P₀ = {max_momentum_error:.3e}")
            if self.adaptive_dt:
                self.logger.info(f"Timestep adaptations: {self.diagnostics['adaptation_count']}")
                self.logger.info(f"Final timestep: dt = {dt_current:.6f}")
            
            if self.diagnostics['warnings']:
                self.logger.warning(f"{len(self.diagnostics['warnings'])} warnings occurred")
        
        return {
            'x': self.x,
            't': t_out,
            'phi': phi_hist,
            'energy': energy_hist,
            'momentum': momentum_hist,
            'energy_error': max_energy_error,
            'momentum_error': max_momentum_error,
            'energy_initial': energy_initial,
            'momentum_initial': momentum_initial,
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
                'energy_tol': self.energy_tol,
                **pot_params
            }
        }
