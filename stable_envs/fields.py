"""Read-only attribute access to the named fields of a State."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDictBase

from ._core.state_schema import get_field, schema_of

if TYPE_CHECKING:
    from .env import Env

__all__ = ["Fields"]


def _names(state: TensorDictBase) -> list[tuple[str, ...]]:
    schema = schema_of(state)
    if schema is not None:
        return [tuple(name.split("/")) for name in schema.names]
    return [key if isinstance(key, tuple) else (key,) for key in state.keys(True, True)]


def _contains(name: tuple[str, ...], chain: tuple[str, ...]) -> bool:
    """Whether ``chain`` is a run of parts of ``name`` that stops before its last part."""
    n = len(chain)
    return any(name[i : i + n] == chain for i in range(len(name) - n))


class Fields:
    """The named fields of a State as attributes, such as ``Fields(state, env).agent.position``.

    A chain of names reads the one field whose name ends with those parts, at part boundaries: ``agent.position``
    reads ``current/body/agent/position`` when no other field name ends with ``agent/position``. A field whose full
    name is the chain is read even when other names end with it. A chain that ends no field name, but that is a
    variation leaf of ``env``, such as ``wall.axis``, reads that leaf from the variation that ``env`` has installed.
    A chain that is only a part of names gives a ``Fields`` for the next part. ``fields["agent.position"]`` reads a
    dotted chain.

    Values have the batch axes first and are the values that ``snv.get_field`` returns, so they can be views of
    ``state``. ``Fields`` has no write operations.

    Args:
        state: A State from ``reset``, ``step`` or ``get_state``.
        env: The environment of ``state``, for its variation leaves; None reads only State fields.

    Raises:
        AttributeError: If an attribute chain names no field, no variation leaf and no part of a name.
        KeyError: If an item chain names no field, no variation leaf and no part of a name.
        ValueError: If a chain ends more than one field name; the message lists them. Also, if a variation leaf
            is read for a State whose batch size is not ``(env.num_envs,)``.
    """

    __slots__ = ("_state", "_env", "_names", "_leaves", "_chain")
    _state: TensorDictBase
    _env: Env | None
    _names: list[tuple[str, ...]]
    _leaves: dict[tuple[str, ...], tuple[int, int]]
    _chain: tuple[str, ...]

    def __init__(self, state: TensorDictBase, env: Env | None = None):
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "_names", _names(state))
        layout = () if env is None else env.variation_space.flat_layout()
        object.__setattr__(self, "_leaves", {tuple(path.split(".")): (off, w) for path, off, w in layout})
        object.__setattr__(self, "_chain", ())

    def _child(self, chain: tuple[str, ...]) -> Fields:
        child = object.__new__(Fields)
        for slot in ("_state", "_env", "_names", "_leaves"):
            object.__setattr__(child, slot, getattr(self, slot))
        object.__setattr__(child, "_chain", chain)
        return child

    def _variation(self, chain: tuple[str, ...]) -> torch.Tensor:
        env = self._env
        assert env is not None
        if tuple(self._state.batch_size) != (env.num_envs,):
            raise ValueError(
                f"variation leaf {'.'.join(chain)!r} comes from the {env.num_envs} installed rows of the "
                f"environment, but the State has batch size {tuple(self._state.batch_size)}"
            )
        off, width = self._leaves[chain]
        node = env.variation_space
        for part in chain:
            node = node[part]
        rows = env.get_variation()[:, off : off + width]
        return rows.reshape(rows.shape[0], *getattr(node, "shape", ()))

    def _resolve(self, chain: tuple[str, ...]):
        ends = [name for name in self._names if name[-len(chain) :] == chain]
        if len(ends) > 1:
            ends = [name for name in ends if name == chain] or ends
        if len(ends) > 1:
            listed = ", ".join("/".join(name) for name in ends)
            raise ValueError(f"State field {'.'.join(chain)!r} is ambiguous; it ends {listed}")
        if ends:
            return get_field(self._state, "/".join(ends[0]))
        if chain in self._leaves:
            return self._variation(chain)
        n = len(chain)
        if any(_contains(name, chain) for name in self._names) or any(
            len(leaf) > n and leaf[:n] == chain for leaf in self._leaves
        ):
            return self._child(chain)
        raise KeyError(f"State has no field and no variation leaf {'.'.join(chain)!r}")

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._resolve((*self._chain, name))
        except KeyError as error:
            raise AttributeError(error.args[0]) from None

    def __getitem__(self, key: str):
        return self._resolve((*self._chain, *key.split(".")))

    def __setattr__(self, name: str, value) -> None:
        raise AttributeError("Fields is read-only; write a field with snv.set_field")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Fields is read-only; write a field with snv.set_field")

    def __dir__(self) -> list[str]:
        n = len(self._chain)
        out = {name[i + n] for name in self._names for i in range(len(name) - n) if name[i : i + n] == self._chain}
        out |= {leaf[n] for leaf in self._leaves if len(leaf) > n and leaf[:n] == self._chain}
        return sorted(out)

    def __repr__(self) -> str:
        return f"Fields({'.'.join(self._chain) or '<root>'})"
