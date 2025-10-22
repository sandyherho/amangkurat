"""Performance timer."""

import time
from contextlib import contextmanager


class Timer:
    """Timer for performance tracking."""
    
    def __init__(self):
        self.times = {}
        self.start_times = {}
    
    def start(self, name: str):
        self.start_times[name] = time.time()
    
    def stop(self, name: str) -> float:
        if name in self.start_times:
            elapsed = time.time() - self.start_times[name]
            self.times[name] = elapsed
            del self.start_times[name]
            return elapsed
        return 0.0
    
    @contextmanager
    def time_section(self, name: str):
        self.start(name)
        yield
        self.stop(name)
    
    def get_times(self) -> dict:
        return self.times.copy()
