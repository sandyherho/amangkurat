import time
import numpy as np
from tqdm import tqdm  # Recommended: pip install tqdm (for progress bar)

# Import both solvers: the original and the optimized version
# Ensure you have created 'amangkurat/core/solver_optimized.py'
from amangkurat.core.solver import KGSolver as OriginalSolver
try:
    from amangkurat.core.solver_optimized import KGSolver as OptimizedSolver
except ImportError:
    raise ImportError("Could not import 'solver_optimized'. Please copy 'solver.py' to 'solver_optimized.py' and apply your changes there.")

from amangkurat.core.initial_conditions import GaussianIC

def run_simulation(solver_class, nx, t_final):
    """
    Helper function to instantiate and run a simulation with a specific solver class.
    """
    # Initialize Solver (quiet mode)
    solver = solver_class(nx=nx, x_min=-30.0, x_max=30.0, adaptive_dt=False, verbose=False)
    
    # Initial Conditions
    ic = GaussianIC(amplitude=1.0, width=2.0, position=0.0)
    phi0, phi_dot0 = ic(solver.x)
    
    # Measure Time
    start_time = time.perf_counter()
    
    result = solver.solve(
        phi0=phi0, 
        phi_dot0=phi_dot0,
        dt=0.005, 
        t_final=t_final,
        potential='linear',
        save_netcdf=False, 
        save_animation=False,
        quiet=True
    )
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    # Return the final field state and the duration
    return result['phi'][-1], duration

def main():
    # --- CONFIGURATION ---
    # NOTE: If running 100 times, you might want to reduce NX or T_FINAL slightly
    # to avoid waiting for hours, unless your optimization is extremely fast.
    ITERATIONS = 100
    NX = 2048
    T_FINAL = 50.0
    
    print(f"--- BENCHMARK SUITE: {ITERATIONS} Iterations ---")
    print(f"Grid Size (NX): {NX} | T_Final: {T_FINAL}")
    print("Comparing: OriginalSolver vs. OptimizedSolver\n")
    
    times_original = []
    times_optimized = []
    
    # Create a loop for the number of iterations
    # We use tqdm if available for a nice progress bar, otherwise standard range
    try:
        iterator = tqdm(range(ITERATIONS), desc="Running Benchmarks", unit="run")
    except ImportError:
        print("Tip: Install 'tqdm' for a progress bar.")
        iterator = range(ITERATIONS)
    
    print(">> Warming up JIT compiler...")
    run_simulation(OriginalSolver, NX, 2.0) 
    run_simulation(OptimizedSolver, NX, 2.0)
    print(">> Warm-up complete. Starting measurements.\n")

    for i in iterator:
        # 1. Run Original
        phi_orig, t_orig = run_simulation(OriginalSolver, NX, T_FINAL)
        times_original.append(t_orig)
        
        # 2. Run Optimized
        phi_opt, t_opt = run_simulation(OptimizedSolver, NX, T_FINAL)
        times_optimized.append(t_opt)
        
        # 3. Verify Correctness IMMEDIATELY
        # We check if the maximum difference is acceptable (almost zero)
        diff = np.max(np.abs(phi_orig - phi_opt))
        
        # If divergence occurs, stop immediately to save time debugging
        if diff > 1e-10:
            print(f"\n[!] CRITICAL ERROR at Iteration {i+1}")
            print(f"    The results diverged! Max diff: {diff:.8e}")
            print("    Stopping benchmark to prevent invalid statistics.")
            return

    # --- STATISTICS REPORT ---
    avg_orig = np.mean(times_original)
    std_orig = np.std(times_original)
    
    avg_opt = np.mean(times_optimized)
    std_opt = np.std(times_optimized)
    
    speedup = avg_orig / avg_opt
    percent_gain = (avg_orig - avg_opt) / avg_orig * 100
    
    print("\n" + "="*40)
    print("             FINAL RESULTS              ")
    print("="*40)
    print(f"Successful Iterations: {ITERATIONS}")
    print(f"Numerical Check:       PASSED (Identical Results)")
    print("-" * 40)
    print(f"ORIGINAL Code:")
    print(f"  Avg Time: {avg_orig:.4f} s  (± {std_orig:.4f})")
    print("-" * 40)
    print(f"OPTIMIZED Code:")
    print(f"  Avg Time: {avg_opt:.4f} s  (± {std_opt:.4f})")
    print("-" * 40)
    print(f"SPEEDUP FACTOR: {speedup:.2f}x")
    print(f"PERFORMANCE GAIN: {percent_gain:.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()