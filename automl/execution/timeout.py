"""
MicroAutoML-Agent — Execution Guards

Wraps long-running execution blocks (e.g. model training) with strict
process-level boundaries to enforce timeouts and RAM limits.
"""

from __future__ import annotations

import multiprocessing
import os
import psutil
import time
import queue
import traceback
from typing import Callable, Any


class TimeoutException(Exception):
    """Raised when an operation exceeds its time budget."""
    pass


def _worker(q: multiprocessing.Queue, func: Callable, args: tuple, kwargs: dict) -> None:
    """Entry point for the subprocess."""
    try:
        # Note: If environment variables (OMP_NUM_THREADS, etc) need to be set,
        # the caller must set them in os.environ *before* starting the process,
        # or we could accept an env dict here. But multiprocessing on fork/spawn 
        # usually inherits the parent env or we can set it in the child.
        result = func(*args, **kwargs)
        q.put(("success", result))
    except Exception as e:
        # We must return the exception string or traceback because arbitrary exceptions
        # may not be cleanly picklable across the queue if they depend on local imports.
        q.put(("error", (type(e).__name__, str(e), traceback.format_exc())))


def run_with_guards(
    func: Callable, 
    args: tuple = (), 
    kwargs: dict | None = None, 
    timeout_seconds: float = 300,
    max_memory_mb: float = float('inf')
) -> Any:
    """
    Executes a function in a subprocess to strictly enforce time and RAM limits.
    
    If limits are exceeded, the child process is forcefully terminated (SIGKILL/TerminateProcess).
    This guarantees that runaway C-extensions (like XGBoost or sklearn trees) die.
    """
    kwargs = kwargs or {}
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_worker, args=(q, func, args, kwargs))
    
    start_time = time.time()
    p.start()
    
    try:
        while p.is_alive():
            elapsed = time.time() - start_time
            
            # 1. Timeout Check
            if elapsed > timeout_seconds:
                p.terminate()
                p.join(timeout=1)
                if p.is_alive():
                    p.kill() # Hard kill if terminate fails
                raise TimeoutException(f"Execution exceeded {timeout_seconds}s timeout.")
                
            # 2. RAM Check
            try:
                # psutil.Process can raise NoSuchProcess if it just died
                proc = psutil.Process(p.pid)
                # Use RSS (Resident Set Size) as the physical memory usage
                mem_mb = proc.memory_info().rss / (1024 * 1024)
                if mem_mb > max_memory_mb:
                    p.terminate()
                    p.join(timeout=1)
                    if p.is_alive():
                        p.kill()
                    raise MemoryError(f"Execution exceeded memory limit ({mem_mb:.1f} MB > {max_memory_mb} MB).")
            except psutil.NoSuchProcess:
                pass  # Process finished or died in the tiny window
                
            time.sleep(0.1)  # Polling interval
            
        # Process has finished
        p.join()
        
        # Check queue for result
        try:
            status, payload = q.get_nowait()
            if status == "success":
                return payload
            else:
                # payload is (err_type, err_msg, err_tb)
                err_type, err_msg, err_tb = payload
                # We raise a generic RuntimeError with the original traceback embedded,
                # which the runner's _classify_error can still parse since it inspects the string.
                raise RuntimeError(f"{err_type}: {err_msg}\nTraceback:\n{err_tb}")
        except queue.Empty as e:
            # If the process exited without putting anything in the queue, it likely crashed
            # due to a segfault, OOM killer, etc.
            exitcode = p.exitcode
            raise RuntimeError(f"Subprocess exited unexpectedly with code {exitcode}.") from e
            
    finally:
        # Cleanup
        if p.is_alive():
            p.terminate()
            p.join()
