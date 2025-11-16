"""amangkurat: Idealized (1+1)D Nonlinear Klein-Gordon Solver"""

__version__ = "0.0.1"
__author__ = "Sandy H. S. Herho"
__license__ = "MIT"

from .core.solver import KGSolver
from .core.initial_conditions import (
    GaussianIC, KinkIC, BreatherIC, KinkAntikinkIC
)
from .io.config_manager import ConfigManager
from .io.data_handler import DataHandler

__all__ = [
    "KGSolver",
    "GaussianIC",
    "KinkIC", 
    "BreatherIC",
    "KinkAntikinkIC",
    "ConfigManager",
    "DataHandler"
]
