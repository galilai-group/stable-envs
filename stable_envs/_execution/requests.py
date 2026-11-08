import time

import numpy as np


def lane_slices(num_envs, processes):
    """Split ``num_envs`` lanes into ``processes`` contiguous, nearly equal slices."""
    return [slice(int(lanes[0]), int(lanes[-1]) + 1) for lanes in np.array_split(np.arange(num_envs), processes)]


class ProcessRequests:
    """Manage one pending request across persistent simulator workers.

    Families supply connections, processes, payloads, and shared buffers. Partial
    dispatch or response failure terminates all owners; a mixed result is never
    returned. Receive and copy shared outputs before starting the next request.
    """

    _worker_family = "Simulation"

    def _responses(self, timeout=None):
        if timeout is None:
            from .. import tuning

            timeout = tuning.get("worker_timeout")
        results, errors = [], []
        deadline = time.monotonic() + timeout
        for conn, process in zip(self._connections, self._processes):
            while not conn.poll(0.05):
                if not process.is_alive() or time.monotonic() >= deadline:
                    raise RuntimeError(f"{self._worker_family} worker {process.pid} exited or timed out")
            ok, value = conn.recv()
            results.append(value)
            if not ok:
                errors.append(value)
        if errors:
            raise RuntimeError(f"{self._worker_family} worker failed:\n" + "\n".join(errors))
        return results

    def _spawn_workers(self, context, target, cpus, args):
        """Start one ``target`` worker per lane slice and store each ready reply in ``worker_info``.

        ``args(rows, cpus)`` returns the arguments after the worker's pipe end, with
        ``cpus`` split evenly across workers. Callers abort on failure.
        """
        processes = len(self._slices)
        groups = [None] * processes if cpus is None else [g.tolist() for g in np.array_split(cpus, processes)]
        for rows, group in zip(self._slices, groups, strict=True):
            parent, child = context.Pipe()
            process = context.Process(target=target, args=(child, *args(rows, group)))
            try:
                process.start()
            except BaseException:
                parent.close()
                raise
            finally:
                child.close()
            self._connections.append(parent)
            self._processes.append(process)
        from .. import tuning

        self.worker_info = self._responses(timeout=max(300, tuning.get("worker_timeout")))

    def _check_idle(self):
        self._check_open()
        if self._pending is not None:
            raise RuntimeError(f"{self._worker_family} process request already in flight")

    def _submit(self, command, payloads=None, order=None):
        """Send ``payloads[i]`` to worker ``i``, in the worker order ``order`` (default: index order)."""
        self._check_idle()
        payloads = [None] * len(self._connections) if payloads is None else payloads
        try:
            for i in range(len(self._connections)) if order is None else order:
                self._connections[i].send((command, payloads[i]))
            self._pending = command
        except BaseException:
            # Abort all owners after partial dispatch; their states may already have diverged.
            self._abort()
            raise

    def _receive(self):
        self._check_open()
        if self._pending is None:
            raise RuntimeError(f"No {self._worker_family} process request in flight")
        try:
            result = self._responses()
            self._pending = None
            return result
        except BaseException:
            self._abort()
            raise

    def _request(self, command, payloads=None):
        self._submit(command, payloads)
        return self._receive()

    def _abort(self):
        for process in self._processes:
            if process.is_alive():
                process.terminate()
        for process in self._processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        for conn in self._connections:
            conn.close()
        self._closed = True
        self._pending = None

    def close(self):
        if not self._closed:
            try:
                if self._pending is None:
                    self._request("close")
                    for process in self._processes:
                        process.join(timeout=5)
            finally:
                self._abort()
