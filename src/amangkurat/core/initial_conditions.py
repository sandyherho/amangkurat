"""Initial conditions for Klein-Gordon equation."""

import numpy as np
from typing import Tuple
import warnings


class GaussianIC:
    """Gaussian wave packet for linear case.
    
    φ(x,0) = A exp[-(x-x₀)²/w²]
    φ̇(x,0) = v·φ(x,0)  (uniform velocity)
    """
    
    def __init__(self, amplitude: float = 1.0, width: float = 2.0,
                 position: float = 0.0, velocity: float = 0.0):
        self.amplitude = amplitude
        self.width = width
        self.position = position
        self.velocity = velocity
        
        if width <= 0:
            raise ValueError(f"Width must be positive, got {width}")
        if abs(amplitude) > 100:
            warnings.warn(f"Large amplitude {amplitude} may cause instability")
    
    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (φ, φ̇) at t=0."""
        xi = (x - self.position) / self.width
        phi = self.amplitude * np.exp(-xi**2)
        phi_dot = self.velocity * phi
        
        return phi, phi_dot


class KinkIC:
    """Kink solution for φ⁴ theory.
    
    Static kink: φ(x) = v·tanh[(x-x₀)/√2]
    Moving kink: Apply Lorentz boost with γ = 1/√(1-v²)
    
    For φ⁴ theory: V(φ) = (λ/4)(φ²-v²)²
    The kink interpolates between vacua at φ = ±v.
    Kink width: ξ = √2
    Kink energy: E = (2√2/3)λv³
    """
    
    def __init__(self, vacuum: float = 1.0, position: float = 0.0,
                 velocity: float = 0.0):
        self.vacuum = vacuum
        self.position = position
        self.velocity = velocity
        
        if abs(velocity) >= 1.0:
            raise ValueError(f"Velocity must be |v| < 1 (speed of light), got {velocity}")
        if vacuum <= 0:
            raise ValueError(f"Vacuum expectation value must be positive, got {vacuum}")
    
    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (φ, φ̇) at t=0."""
        
        if abs(self.velocity) < 1e-10:
            # Static kink
            xi = (x - self.position) / np.sqrt(2)
            phi = self.vacuum * np.tanh(xi)
            phi_dot = np.zeros_like(x)
        else:
            # Moving kink with Lorentz boost
            gamma = 1.0 / np.sqrt(1 - self.velocity**2)
            
            # Boosted coordinate: ξ = γ(x - vt - x₀) at t=0
            xi = gamma * (x - self.position) / np.sqrt(2)
            
            # Field profile
            phi = self.vacuum * np.tanh(xi)
            
            # Velocity profile: φ̇ = -v·∂ₓφ for right-moving kink
            # ∂ₓφ = (v/√2)γ sech²(ξ)
            phi_dot = -(self.velocity * self.vacuum * gamma / np.sqrt(2)) * (1.0 / np.cosh(xi))**2
        
        return phi, phi_dot


class BreatherIC:
    """Breather solution for sine-Gordon equation.
    
    
    Exact breather solution:
    φ(x,t) = 4·arctan[sin(ωt) / (ω_x cosh(ω_x·x))]
    
    where ω_x = √(1 - ω²) and 0 < ω < 1
    
    At t=0:
    φ(x,0) = 4·arctan[sin(0) / (ω_x cosh(ω_x·x))] = 0  (problematic!)
    
    Better: Start at t = π/(2ω) where sin(ωt) = 1:
    φ(x,0) = 4·arctan[1 / (ω_x cosh(ω_x·x))]
    φ̇(x,0) = -4ω·ω_x sinh(ω_x·x) / [cosh(ω_x·x) + 1/ω_x²]
    
    OLD BUG: The original code had sin(0) = 0 hardcoded!
    """
    
    def __init__(self, frequency: float = 0.5, position: float = 0.0):
        self.omega = frequency
        self.position = position
        
        if not (0 < frequency < 1):
            raise ValueError(f"Breather frequency must be 0 < ω < 1, got {frequency}")
        
        self.omega_x = np.sqrt(1 - frequency**2)
    
    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (φ, φ̇) at t=0, starting at maximum amplitude."""
        
        xi = self.omega_x * (x - self.position)
        cosh_xi = np.cosh(xi)
        sinh_xi = np.sinh(xi)
        
        # Start at t_0 = π/(2ω) where sin(ωt) = 1 (maximum amplitude)
        # This gives the breather at its most visible state
        
        # Field: φ = 4·arctan[1 / (ω_x·cosh(ω_x·x))]
        phi = 4.0 * np.arctan(1.0 / (self.omega_x * cosh_xi))
        
        # Time derivative at maximum amplitude
        # φ̇ = ∂ₜφ|_{sin(ωt)=1} = -4ω·ω_x sinh(ω_x·x) / [ω_x²·cosh²(ω_x·x) + 1]
        denominator = self.omega_x**2 * cosh_xi**2 + 1.0
        phi_dot = -4.0 * self.omega * self.omega_x * sinh_xi / denominator
        
        return phi, phi_dot


class KinkAntikinkIC:
    """Kink-antikink pair for collision studies in φ⁴ theory.
    
    Configuration:
    - Kink at x = -d/2 moving right with velocity +v
    - Antikink at x = +d/2 moving left with velocity -v
    
    Both are Lorentz boosted to have initial velocities.
    
    Physics: When they collide, they may:
    1. Annihilate into radiation
    2. Bounce back
    3. Form a breather-like bound state
    
    The outcome depends on the collision velocity v.
    """
    
    def __init__(self, vacuum: float = 1.0, separation: float = 20.0,
                 velocity: float = 0.3):
        self.vacuum = vacuum
        self.separation = separation
        self.velocity = velocity
        
        if abs(velocity) >= 1.0:
            raise ValueError(f"Velocity must be |v| < 1, got {velocity}")
        if separation <= 0:
            raise ValueError(f"Separation must be positive, got {separation}")
        
        # Ensure separation is large enough to avoid initial overlap
        kink_width = np.sqrt(2)
        if separation < 4 * kink_width:
            warnings.warn(f"Separation {separation:.2f} is small compared to "
                        f"kink width {kink_width:.2f}. Initial overlap may occur.")
    
    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (φ, φ̇) at t=0."""
        
        gamma = 1.0 / np.sqrt(1 - self.velocity**2)
        
        # Kink position (left, moving right)
        x1 = -self.separation / 2
        xi1 = gamma * (x - x1) / np.sqrt(2)
        kink = self.vacuum * np.tanh(xi1)
        kink_dot = -(self.velocity * self.vacuum * gamma / np.sqrt(2)) * (1.0 / np.cosh(xi1))**2
        
        # Antikink position (right, moving left)
        x2 = self.separation / 2
        xi2 = gamma * (x - x2) / np.sqrt(2)
        antikink = -self.vacuum * np.tanh(xi2)
        antikink_dot = (self.velocity * self.vacuum * gamma / np.sqrt(2)) * (1.0 / np.cosh(xi2))**2
        
        # Superpose the configurations
        # This is approximate - true multi-soliton solution is more complex
        phi = kink + antikink
        phi_dot = kink_dot + antikink_dot
        
        # Check for excessive initial overlap
        max_phi = np.max(np.abs(phi))
        if max_phi > 1.5 * self.vacuum:
            warnings.warn(f"Large initial field value {max_phi:.2f} > {1.5*self.vacuum:.2f}. "
                        f"Consider increasing separation.")
        
        return phi, phi_dot


# Validation function
def validate_initial_condition(phi: np.ndarray, phi_dot: np.ndarray, 
                              x: np.ndarray) -> None:
    """Check initial conditions for common problems."""
    
    # Check for NaN or Inf
    if not np.isfinite(phi).all() or not np.isfinite(phi_dot).all():
        raise ValueError("Initial condition contains NaN or Inf")
    
    # Check for reasonable magnitudes
    max_phi = np.max(np.abs(phi))
    max_phi_dot = np.max(np.abs(phi_dot))
    
    if max_phi > 1000:
        warnings.warn(f"Large initial field |φ| = {max_phi:.2f} may cause instability")
    
    if max_phi_dot > 1000:
        warnings.warn(f"Large initial velocity |φ̇| = {max_phi_dot:.2f} may cause instability")
    
    # Check for constant field (boring)
    if np.allclose(phi, phi[0], atol=1e-10):
        warnings.warn("Initial field is constant (trivial solution)")
