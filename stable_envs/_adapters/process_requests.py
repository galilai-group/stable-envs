import time


class ProcessRequests:
    """Manage one pending request across persistent simulator workers.

    Families supply connections, processes, payloads, and shared buffers. Partial
    dispatch or response failure terminates all owners; a mixed result is never
    returned. Receive and copy shared outputs before starting the next request.
    """

    _worker_family = "Simulation"

    def _responses(self, timeout=120):
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

    def _check_idle(self):
        self._check_open()
        if self._pending is not None:
            raise RuntimeError(f"{self._worker_family} process request already in flight")

    def _submit(self, command, payloads=None):
        self._check_idle()
        payloads = [None] * len(self._connections) if payloads is None else payloads
        try:
            for conn, payload in zip(self._connections, payloads):
                conn.send((command, payload))
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
