"""Guarded ordinary DMC rounds with one device-result readback."""

from functools import cache

import torch

wp = None


class PairFailure(RuntimeError):
    """A failed pair, with the last successful round's borrowed checkpoint."""

    def __init__(self, message, *, completed_rounds, checkpoint=None):
        super().__init__(message)
        self.completed_rounds = completed_rounds
        self.checkpoint = checkpoint


class _RetainedScratch:
    """Retain each warm-up allocation for reuse while recording the graph.

    Conditional CUDA graph bodies cannot contain allocation/free nodes. Warp's
    public allocator scope lets its physics functions use preallocated storage
    without changing those functions. Capture must request the same sizes in
    the same order; mismatches fail rather than silently running another path.
    """

    def __init__(self, device):
        self.device = device
        self.buffers, self.sizes = [], []
        self.recording = True
        self.cursor = 0

    def allocate(self, size_in_bytes):
        if self.recording:
            buffer = torch.empty(size_in_bytes, dtype=torch.uint8, device=self.device)
            self.buffers.append(buffer)
            self.sizes.append(size_in_bytes)
            return buffer.data_ptr()
        if self.cursor >= len(self.buffers):
            raise RuntimeError("DMC MJWarp paired capture requested unexpected scratch storage")
        if size_in_bytes != self.sizes[self.cursor]:
            raise RuntimeError("DMC MJWarp paired capture changed a scratch allocation's size")
        pointer = self.buffers[self.cursor].data_ptr()
        self.cursor += 1
        return pointer

    def deallocate(self, ptr, size_in_bytes):
        # The graph owns these buffers through this allocator's retained list.
        pass

    def begin_capture(self):
        self.cursor = 0
        self.recording = False

    def finish_capture(self):
        if self.cursor != len(self.buffers):
            raise RuntimeError("DMC MJWarp paired capture omitted a recorded scratch allocation")

    def close(self):
        self.buffers.clear()
        self.sizes.clear()


@cache
def _pack_result_kernel():
    @wp.kernel
    def pack(error: wp.array[int], round_index: int, status: wp.array[int], continuing: wp.array[int]):
        # Each round currently uses seven bits: capacity plus six finite checks.
        # Keep the first result intact when the second round records its result.
        if round_index == 1:
            status[0] = error[0]
        else:
            status[0] = status[0] | (error[0] << 8)
        if error[0] == 0:
            continuing[0] = 1
        else:
            continuing[0] = 0

    return pack


class PairedRound:
    """Two ordinary physics rounds with a device guard before the second.

    The adapter chooses this path only when neither round needs a native reset.
    Public lifecycle, time, and cumulative reward remain adapter policy. Graph
    outputs and the first-round checkpoint are borrowed until the next replay.
    """

    def __init__(self, runtime, task):
        global wp
        if wp is None:
            import warp

            wp = warp
        self.runtime, self.task = runtime, task
        self.graph = None
        self.first_reward = self.outputs = None
        self.status = torch.zeros(1, dtype=torch.int32, device=runtime.device)
        self._continuing = torch.zeros_like(self.status)
        self._status_wp = wp.from_torch(self.status)
        self._continuing_wp = wp.from_torch(self._continuing)
        self.checkpoint = {
            key: torch.empty_like(runtime._data_view(name), dtype=torch.float64)
            for key, name in runtime._physical.items()
            if key != "time"
        }
        self._scratch = _RetainedScratch(runtime.device)
        self._capture()

    def _record_round(self, round_index):
        self.runtime._record_result()
        wp.launch(
            _pack_result_kernel(),
            dim=1,
            inputs=[self.runtime._result_error, round_index],
            outputs=[self._status_wp, self._continuing_wp],
            device=self.runtime._wp_device,
        )

    def _second_round(self):
        # Copy only after the first round succeeds. A second-round failure can
        # then commit this checkpoint without exposing invalid physics.
        self.runtime.copy_physical_state(self.checkpoint, include_time=False)
        self.runtime._advance()
        self._record_round(2)

    def _capture(self):
        runtime = self.runtime
        original = runtime.physical_state()
        controls = runtime._data_view("ctrl").clone()

        def restore():
            runtime._load(original)
            runtime._data_view("ctrl").copy_(controls)

        try:
            # Warm both round variants from the original checkpoint, restoring
            # between them. Setup never commits either speculative checkpoint.
            for round_index in (1, 2):
                with wp.ScopedAllocator(runtime._wp_device, self._scratch):
                    runtime._advance()
                    self._record_round(round_index)
                if round_index == 1:
                    runtime.copy_physical_state(self.checkpoint, include_time=False)
                runtime._task_outputs(self.task, reward_only=round_index == 1)
                runtime._torch_stream.synchronize()
                restore()
            self._scratch.begin_capture()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=runtime._torch_stream):
                wp.capture_begin(external=True, force_module_load=False, capture_mode=wp.CaptureMode.GLOBAL)
                try:
                    # Both rounds use retained Warp scratch. In particular,
                    # the conditional body's graph contains no allocation nodes.
                    with wp.ScopedAllocator(runtime._wp_device, self._scratch):
                        runtime._advance()
                        self._record_round(1)
                        self.first_reward = runtime._task_outputs(self.task, reward_only=True)
                        wp.capture_if(self._continuing_wp, on_true=self._second_round)
                        # Task formulas are side-effect-free and may run even
                        # when a round failed; their outputs are then discarded.
                        self.outputs = runtime._task_outputs(self.task, reward_only=False)
                finally:
                    wp.capture_end()
            self._scratch.finish_capture()
            self.graph = graph
        finally:
            restore()

    def replay(self):
        self.graph.replay()
        # Exactly one read of the packed device result after the guarded pair.
        status = int(self.status[0])
        first, second = status & 255, (status >> 8) & 255
        if first:
            raise PairFailure(self.runtime._result_message(first), completed_rounds=0)
        if second:
            raise PairFailure(self.runtime._result_message(second), completed_rounds=1, checkpoint=self.checkpoint)
        return self.first_reward, self.outputs

    def close(self):
        # The runtime synchronizes its stream before releasing captured storage.
        self.graph = None
        self.first_reward = self.outputs = None
        self.checkpoint.clear()
        self._status_wp = self._continuing_wp = None
        self.status = self._continuing = None
        self._scratch.close()
