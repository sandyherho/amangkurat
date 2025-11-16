#!/usr/bin/env python
"""Command Line Interface for amangkurat Klein-Gordon Solver."""

import argparse
import sys
import numpy as np
from pathlib import Path

from .core.solver import KGSolver
from .core.initial_conditions import (
    GaussianIC, KinkIC, BreatherIC, KinkAntikinkIC
)
from .io.config_manager import ConfigManager
from .io.data_handler import DataHandler
from .visualization.animator import Animator
from .utils.logger import SimulationLogger
from .utils.timer import Timer


def print_header():
    print("\n" + "=" * 70)
    print(" " * 15 + "amangkurat: Idealized Klein-Gordon Solver")
    print(" " * 25 + "Version 0.0.1")
    print("=" * 70)
    print("\n  Nonlinear Klein-Gordon Equation Solver")
    print("  Symplectic Methods + Spectral Accuracy")
    print("\n  Author: Sandy H. S. Herho")
    print("  License: MIT")
    print("=" * 70 + "\n")


def normalize_name(name: str) -> str:
    clean = name.lower().replace(' - ', '_').replace('-', '_').replace(' ', '_')
    while '__' in clean:
        clean = clean.replace('__', '_')
    return clean.rstrip('_')


def create_initial_condition(config: dict, x: np.ndarray):
    ic_type = config.get('ic_type', 'gaussian')
    
    if ic_type == 'gaussian':
        ic = GaussianIC(
            amplitude=config.get('amplitude', 1.0),
            width=config.get('width', 2.0),
            position=config.get('position', 0.0),
            velocity=config.get('velocity', 0.0)
        )
    elif ic_type == 'kink':
        ic = KinkIC(
            vacuum=config.get('vacuum', 1.0),
            position=config.get('position', 0.0),
            velocity=config.get('velocity', 0.0)
        )
    elif ic_type == 'breather':
        ic = BreatherIC(
            frequency=config.get('frequency', 0.5),
            position=config.get('position', 0.0)
        )
    elif ic_type == 'kink_antikink':
        ic = KinkAntikinkIC(
            vacuum=config.get('vacuum', 1.0),
            separation=config.get('separation', 20.0),
            velocity=config.get('velocity', 0.3)
        )
    else:
        raise ValueError(f"Unknown IC type: {ic_type}")
    
    return ic(x)


def run_scenario(config: dict, output_dir: str = "outputs",
                verbose: bool = True, n_cores: int = None):
    scenario_name = config.get('scenario_name', 'simulation')
    clean_name = normalize_name(scenario_name)
    
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'=' * 60}")
    
    logger = SimulationLogger(clean_name, "logs", verbose)
    timer = Timer()
    timer.start("total")
    
    try:
        logger.log_parameters(config)
        
        with timer.time_section("solver_init"):
            if verbose:
                print("\n[1/4] Initializing solver...")

            solver = KGSolver(
                nx=config.get('nx', 512),
                x_min=config.get('x_min', -30.0),
                x_max=config.get('x_max', 30.0),
                verbose=verbose,
                logger=logger,
                n_cores=n_cores,
                adaptive_dt=config.get('adaptive_dt', True)
            )
        
        with timer.time_section("initial_condition"):
            if verbose:
                print("\n[2/4] Creating initial condition...")
            
            phi0, phi_dot0 = create_initial_condition(config, solver.x)
        
        with timer.time_section("simulation"):
            if verbose:
                print("\n[3/4] Solving Klein-Gordon equation...")
            
            pot_params = {}
            potential = config.get('potential', 'phi4')
            
            if potential == 'linear':
                pot_params['mass'] = config.get('mass', 1.0)
            elif potential == 'phi4':
                pot_params['lambda'] = config.get('lambda', 1.0)
                pot_params['vacuum'] = config.get('vacuum', 1.0)
            
            result = solver.solve(
                phi0=phi0,
                phi_dot0=phi_dot0,
                dt=config.get('dt', 0.01),
                t_final=config.get('t_final', 50.0),
                potential=potential,
                n_snapshots=config.get('n_frames', 200),
                **pot_params
            )
            
            logger.log_results(result)
        
        if config.get('save_netcdf', True):
            if verbose:
                print("\n[4/4] Saving outputs...")
            
            filename = f"{clean_name}.nc"
            DataHandler.save_netcdf(filename, result, config, output_dir)
            
            if verbose:
                print(f"      NetCDF: {output_dir}/{filename}")
        
        if config.get('save_animation', True):
            filename = f"{clean_name}.gif"
            
            Animator.create_gif(
                result,
                filename,
                output_dir,
                scenario_name,
                fps=config.get('fps', 30),
                dpi=config.get('dpi', 150),
                colormap=config.get('colormap', 'plasma')
            )
        
        timer.stop("total")
        logger.log_timing(timer.get_times())
        
        if verbose:
            total = timer.times.get('total', 0)
            print(f"\n{'=' * 60}")
            print("COMPLETED SUCCESSFULLY")
            print(f"  Total time: {total:.2f} s")
            print(f"{'=' * 60}\n")
    
    except Exception as e:
        logger.error(f"Failed: {str(e)}")
        if verbose:
            print(f"\nERROR: {str(e)}\n")
        raise
    finally:
        logger.finalize()


def main():
    parser = argparse.ArgumentParser(
        description='amangkurat: Klein-Gordon Solver',
        epilog='Example: amangkurat case1 --cores 8'
    )
    
    parser.add_argument(
        'case',
        nargs='?',
        choices=['case1', 'case2', 'case3', 'case4'],
        help='Test case (case1-4)'
    )
    
    parser.add_argument('--config', '-c', type=str,
                       help='Custom config file')
    
    parser.add_argument('--all', '-a', action='store_true',
                       help='Run all cases')
    
    parser.add_argument('--output-dir', '-o', type=str, default='outputs',
                       help='Output directory')
    
    parser.add_argument('--cores', type=int, default=None,
                       help='CPU cores')
    
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet mode')
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    if verbose:
        print_header()
    
    if args.config:
        config = ConfigManager.load(args.config)
        run_scenario(config, args.output_dir, verbose, args.cores)
    
    elif args.all:
        configs_dir = Path(__file__).parent.parent.parent / 'configs'
        config_files = sorted(configs_dir.glob('case*.txt'))
        
        for i, cfg_file in enumerate(config_files, 1):
            if verbose:
                print(f"\n[{i}/{len(config_files)}] {cfg_file.stem}...")
            
            config = ConfigManager.load(str(cfg_file))
            run_scenario(config, args.output_dir, verbose, args.cores)
    
    elif args.case:
        case_map = {
            'case1': 'case1_linear_wave',
            'case2': 'case2_kink',
            'case3': 'case3_breather',
            'case4': 'case4_collision'
        }
        
        cfg_name = case_map[args.case]
        configs_dir = Path(__file__).parent.parent.parent / 'configs'
        cfg_file = configs_dir / f'{cfg_name}.txt'
        
        if cfg_file.exists():
            config = ConfigManager.load(str(cfg_file))
            run_scenario(config, args.output_dir, verbose, args.cores)
        else:
            print(f"ERROR: Config not found: {cfg_file}")
            sys.exit(1)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
