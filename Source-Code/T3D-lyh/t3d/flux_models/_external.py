"""Process and scratch-path utilities for external transport closures."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


def run_workspace(configured_root):
    """Never accept stale outputs from an unrelated T3D driver invocation."""
    root = Path(configured_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.mkdtemp(prefix="t3d-run-", dir=root)


def trial_tag(time):
    return f"t{time.step_idx:02}-p{time.iter_idx}-a{getattr(time, 'retry_generation', 0)}"


def command_prefix(command):
    """Parse an explicit launcher as argv, not as a shell program."""
    return shlex.split(command) if isinstance(command, str) else list(command)


def available_cpus():
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return os.cpu_count() or 1


def worker_count(total, per_job):
    if type(total) is not int or type(per_job) is not int or per_job < 1 or total < per_job:
        raise ValueError("CPU capacity and CPUs per case must be positive integers; capacity must cover a case")
    return total // per_job


def run_in_parallel(function, cases, workers):
    """Bound the number of simultaneous external processes; surface worker errors."""
    if workers < 1:
        raise ValueError("at least one concurrent case is required")
    if not cases:
        return
    if workers == 1:
        for case in cases:
            function(case)
        return
    with ThreadPoolExecutor(max_workers=min(workers, len(cases))) as executor:
        futures = [executor.submit(function, case) for case in cases]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def require_outputs(paths, model):
    missing = [os.fspath(path) for path in paths if not valid_output(path)]
    if missing:
        raise RuntimeError(f"{model} did not produce complete output: {', '.join(missing)}")


def valid_output(path):
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0


def start_job(args, log_path, cwd=None, env=None):
    # Child owns a duplicate of the file descriptor after Popen returns.
    with open(log_path, "w") as log:
        return subprocess.Popen(args, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)


def wait_checked(process, model):
    code = process.wait()
    if code:
        raise RuntimeError(f"{model} exited with status {code}: {process.args}")


def cancel_jobs(processes):
    for process in processes:
        if process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            process.wait()
