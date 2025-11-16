"""Simulation logger."""

import logging
from pathlib import Path


class SimulationLogger:
    """Logger for simulations."""
    
    def __init__(self, scenario_name: str, log_dir: str = "logs",
                 verbose: bool = True):
        self.scenario_name = scenario_name
        self.log_dir = Path(log_dir)
        self.verbose = verbose
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{scenario_name}.log"
        
        self.logger = self._setup_logger()
        self.warnings = []
        self.errors = []
    
    def _setup_logger(self):
        logger = logging.getLogger(f"kg_{self.scenario_name}")
        logger.setLevel(logging.DEBUG)
        logger.handlers = []
        
        handler = logging.FileHandler(self.log_file, mode='w')
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
        self.warnings.append(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)
        self.errors.append(msg)
    
    def log_parameters(self, params: dict):
        self.info("=" * 60)
        self.info(f"PARAMETERS - {params.get('scenario_name', 'Unknown')}")
        self.info("=" * 60)
        for key, value in sorted(params.items()):
            self.info(f"  {key}: {value}")
        self.info("=" * 60)
    
    def log_timing(self, timing: dict):
        self.info("=" * 60)
        self.info("TIMING")
        self.info("=" * 60)
        for key, value in sorted(timing.items()):
            self.info(f"  {key}: {value:.3f} s")
        self.info("=" * 60)
    
    def log_results(self, results: dict):
        self.info("=" * 60)
        self.info("RESULTS")
        self.info("=" * 60)
        
        # Log diagnostics if available
        if 'diagnostics' in results:
            diag = results['diagnostics']
            if 'adaptation_count' in diag:
                self.info(f"  Timestep adaptations: {diag['adaptation_count']}")
            if 'warnings' in diag and diag['warnings']:
                self.info(f"  Warnings: {len(diag['warnings'])}")
        
        # Log final parameters
        if 'params' in results:
            params = results['params']
            if 'n_steps' in params:
                self.info(f"  Total steps: {params['n_steps']}")
            if 'dt_final' in params:
                self.info(f"  Final timestep: {params['dt_final']:.6f}")
        
        self.info("=" * 60)
    
    def finalize(self):
        self.info("=" * 60)
        self.info(f"Completed: {self.scenario_name}")
        self.info("=" * 60)
