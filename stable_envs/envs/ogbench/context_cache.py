"""OGBench context residency, bounded by accounted GPU allocation bytes.

The current batch's keys take precedence over inactive keys. A lease prevents
release while rendering/compositing still uses a context's buffers. Workloads
larger than the budget stream through the cache; an individual leased group
that cannot fit fails explicitly. Construction can temporarily allocate one
additional context before its exact size is known; the limit governs retained
contexts, not all allocations in the process.
"""

import os
import threading
from collections import Counter, OrderedDict
from contextlib import contextmanager
from dataclasses import fields, is_dataclass

from ...render import _resources


class ContextCache:
    def __init__(self, on_evict, memory_info, budget_bytes=None):
        self.name = "ogbench.ctx"
        self.on_evict = on_evict
        self.memory_info = memory_info
        self.budget_bytes = budget_bytes
        self._lock = threading.RLock()
        self._d = OrderedDict()
        self._sizes = {}
        self._known_sizes = {}
        self._budgets = {}
        self._used = Counter()
        self._leases = Counter()
        self._active = set()
        self.hits = self.misses = self.evictions = self.active_evictions = 0

    def budget(self, device):
        if device not in self._budgets:
            configured = self.budget_bytes
            if configured is None:
                configured = os.environ.get("STABLE_ENVS_OGB_CONTEXT_CACHE_BYTES")
            if configured is None:
                free, total = self.memory_info(device)
                configured = min(total // 4, free // 2)
            configured = int(configured)
            if configured <= 0:
                raise ValueError("OGBench context cache budget must be positive bytes")
            self._budgets[device] = configured
        return self._budgets[device]

    def get(self, key, default=None):
        with self._lock:
            if key not in self._d:
                self.misses += 1
                return default
            self.hits += 1
            self._d.move_to_end(key)
            return self._d[key]

    def _room(self, key, size):
        device = key[0]
        budget = self.budget(device)
        # Check before evicting: never destroy a live context to admit another.
        locked = sum(self._sizes[k] for k in self._d if k[0] == device and self._leases[k])
        if size + locked > budget:
            raise MemoryError(
                f"OGBench leased contexts need {size + locked} bytes, above the {budget}-byte context cache budget. "
                "Reduce chunk_size/resolution or increase STABLE_ENVS_OGB_CONTEXT_CACHE_BYTES."
            )
        while self._used[device] + size > budget:
            candidates = [k for k in self._d if k[0] == device and not self._leases[k]]
            victim = next((k for k in candidates if k not in self._active), candidates[0])
            self.active_evictions += victim in self._active
            self.evictions += 1
            _resources._bump(_resources._EVICTED, self.name)
            value = self._d.pop(victim)
            self._used[device] -= self._sizes.pop(victim)
            self.on_evict(value)
            del value

    def prepare(self, key):
        """Evict before allocation when a previously measured size is available."""
        with self._lock:
            self._room(key, self._known_sizes.get(key, 0))

    def put(self, key, value, size):
        with self._lock:
            if key in self._d:
                raise ValueError("Cannot replace a cached GPU context")
            size = int(size)
            if size <= 0:
                raise ValueError("GPU context allocation size must be positive")
            self._room(key, size)
            self._d[key] = value
            self._sizes[key] = size
            self._known_sizes[key] = size
            self._used[key[0]] += size
        return value

    @contextmanager
    def working_set(self, keys):
        with self._lock:
            previous = self._active
            self._active = previous | set(keys)
            try:
                yield
            finally:
                self._active = previous

    @contextmanager
    def lease(self, keys):
        with self._lock:
            keys = tuple(keys)
            self._leases.update(keys)
            try:
                yield
            finally:
                self._leases.subtract(keys)
                self._leases += Counter()  # discard zero counts

    def clear(self):
        with self._lock:
            if any(self._leases.values()):
                raise RuntimeError("Cannot clear leased rendering contexts")
            values = list(self._d.values())
            self._d.clear()
            self._sizes.clear()
            self._used.clear()
            self._known_sizes.clear()
            for value in values:
                self.on_evict(value)

    def keys(self):
        with self._lock:
            return list(self._d)

    def __contains__(self, key):
        return key in self._d

    def __len__(self):
        return len(self._d)

    def stats(self):
        with self._lock:
            return dict(
                hits=self.hits,
                misses=self.misses,
                evictions=self.evictions,
                active_evictions=self.active_evictions,
                entries=len(self._d),
                retained_bytes=dict(self._used),
                budget_bytes=dict(self._budgets),
            )


def allocation_bytes(context):
    """Count unique CUDA buffer spans, including Torch views of Warp allocations.

    Context construction also measures CUDA free-memory change, to account for
    graph/BVH allocations not exposed as arrays. The caller charges the larger
    of that change and this buffer census.
    """
    import torch
    import warp as wp

    seen, spans = set(), []

    def visit(obj):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, wp.array):
            if obj.device.is_cuda and obj.ptr and obj.capacity:
                spans.append((int(obj.ptr), int(obj.ptr) + int(obj.capacity)))
        elif torch.is_tensor(obj):
            if obj.is_cuda:
                storage = obj.untyped_storage()
                spans.append((storage.data_ptr(), storage.data_ptr() + storage.nbytes()))
        elif isinstance(obj, dict):
            for value in obj.values():
                visit(value)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                visit(value)
        elif is_dataclass(obj) and not isinstance(obj, type):
            for field in fields(obj):
                visit(getattr(obj, field.name))
        elif type(obj).__module__.startswith(("mujoco_warp", "warp")) and hasattr(obj, "__dict__"):
            visit(vars(obj))

    for name in ("wm", "wd", "rc", "bgra", "seg", "button_tables"):
        visit(getattr(context, name, None))
    end = total = 0
    for lo, hi in sorted(spans):
        total += max(0, hi - max(lo, end))
        end = max(end, hi)
    return total
