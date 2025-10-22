"""Initial conditions for Klein-Gordon equation."""

import numpy as np
from typing import Tuple


class GaussianIC:
    """Gaussian wave packet for linear case."""
    
    def __init__(self, amplitude: float = 1.0, width: float = 2.0,
                 position: float = 0.0, velocity: float = 0.0):
        self.amplitude = amplitude
        self.width = width
        self.position = position
        self.velocity = velocity
    
    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        phi = self.amplitude * np.exp(-((x - self.position) / self.width)**2)
        phi_dot = self.velocity * phi
        return phi, phi_dot


class KinkIC:
    """Kink solution for φ⁴ theory: φ = v·tanh(x/√2)."""
    
    def __init__(self, vacuum: float = 1.0, position: float = 0.0,
                 velocity: float = 0.0):
        self.vacuum = vacuum
        self.position = position
        self.velocity = velocity
    
    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xi = (x - self.position) / np.sqrt(2)
        phi = self.vacuum * np.tanh(xi)
        
        if abs(self.velocity) < 1e-10:
            phi_dot = np.zeros_like(x)
        else:
            gamma = 1.0 / np.sqrt(1 - self.velocity**2)
            phi_dot = gamma * self.velocity * self.vacuum / (np.cosh(xi)**2 * np.sqrt(2))
        
        return phi, phi_dot


class BreatherIC:
    """Breather solution for sine-Gordon."""
    
    def __init__(self, frequency: float = 0.5, position: float = 0.0):
        self.omega = frequency
        self.position = position
    
    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        omega_x = np.sqrt(1 - self.omega**2)
        xi = x - self.position
        
        phi = 4 * np.arctan(
            np.sin(self.omega * 0) / np.cosh(omega_x * xi)
        )
        phi_dot = np.zeros_like(x)
        
        return phi, phi_dot


class KinkAntikinkIC:
    """Kink-antikink pair for collision studies."""
    
    def __init__(self, vacuum: float = 1.0, separation: float = 20.0,
                 velocity: float = 0.3):
        self.vacuum = vacuum
        self.separation = separation
        self.velocity = velocity
    
    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gamma = 1.0 / np.sqrt(1 - self.velocity**2)
        
        x1 = -self.separation / 2
        x2 = self.separation / 2
        
        kink = self.vacuum * np.tanh(gamma * (x - x1) / np.sqrt(2))
        antikink = -self.vacuum * np.tanh(gamma * (x - x2) / np.sqrt(2))
        
        phi = kink + antikink
        
        phi_dot = (
            gamma * self.velocity * self.vacuum / (np.cosh(gamma * (x - x1) / np.sqrt(2))**2 * np.sqrt(2)) -
            gamma * self.velocity * self.vacuum / (np.cosh(gamma * (x - x2) / np.sqrt(2))**2 * np.sqrt(2))
        )
        
        return phi, phi_dot
