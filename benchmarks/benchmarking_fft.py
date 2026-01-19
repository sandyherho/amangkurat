import numpy as np
import timeit

# Setup
Nx = 4096  # Large grid to highlight the difference
phi = np.random.randn(Nx)
k_fft = np.fft.fftfreq(Nx)
k_rfft = np.fft.rfftfreq(Nx)

def original_fft_method():
    # The old way
    phi_hat = np.fft.fft(phi)
    lap_hat = -(k_fft**2) * phi_hat
    return np.real(np.fft.ifft(lap_hat))

def new_rfft_method():
    # The new way
    phi_hat = np.fft.rfft(phi)
    lap_hat = -(k_rfft**2) * phi_hat
    return np.fft.irfft(lap_hat, n=Nx)

# Warmup (to load libraries)
original_fft_method()
new_rfft_method()

# Benchmark
loops = 1000
time_original = timeit.timeit(original_fft_method, number=loops)
time_new = timeit.timeit(new_rfft_method, number=loops)

print(f"--- Benchmark FFT vs RFFT (Nx={Nx}, {loops} loops) ---")
print(f"Original (FFT) : {time_original:.4f} s")
print(f"New (RFFT)     : {time_new:.4f} s")
print(f"Speedup        : {time_original/time_new:.2f}x faster")