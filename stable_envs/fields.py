"""Read-only attribute access to the named fields of a State."""

from __future__ import annotations

import torch
from tensordict import TensorDictBase

from ._core.state_schema import StateSchema, schema_of

__all__ = ["Fields"]


def _contains(name: tuple[str, ...], chain: tuple[str, ...]) -> bool:
    """Whether ``chain`` is a run of parts of ``name`` that stops before its last part."""
    n = len(chain)
    return any(name[i : i + n] == chain for i in range(len(name) - n))


class Fields:
    """The named fields of a State as attributes, such as ``Fields(state).agent.position``.

    A chain of names reads the one field whose name ends with those parts, at part boundaries: ``agent.position``
    reads ``current/body/agent/position`` when no other field name ends with ``agent/position``. A field whose full
    name is the chain is read even when other names end with it. A chain that is only a part of names gives a
    ``Fields`` for the next part. ``fields["agent.position"]`` and ``fields["agent/position"]`` read a chain.

    Values have the batch axes first and are the values that ``snv.get_field`` returns, so they can be views of
    ``state``. ``Fields`` has no write operations.

    Args:
        state: A State from ``reset``, ``step`` or ``get_state``.

    Raises:
        AttributeError: If an attribute chain names no field and no part of a field name.
        KeyError: If an item chain names no field and no part of a field name.
        ValueError: If a chain ends more than one field name; the message lists them.
    """

    __slots__ = ("_state", "_schema", "_names", "_chain")
    _state: TensorDictBase
    _schema: StateSchema | None
    _names: list[tuple[str, ...]]
    _chain: tuple[str, ...]

    def __init__(self, state: TensorDictBase):
        schema = schema_of(state)
        if schema is not None:
            names = [tuple(name.split("/")) for name in schema.names]
        else:
            names = [key if isinstance(key, tuple) else (key,) for key in state.keys(True, True)]
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_schema", schema)
        object.__setattr__(self, "_names", names)
        object.__setattr__(self, "_chain", ())

    def _child(self, chain: tuple[str, ...]) -> Fields:
        child = object.__new__(Fields)
        for slot in ("_state", "_schema", "_names"):
            object.__setattr__(child, slot, getattr(self, slot))
        object.__setattr__(child, "_chain", chain)
        return child

    def _read(self, name: tuple[str, ...]) -> torch.Tensor:
        if self._schema is not None:
            return self._schema.get(self._state, "/".join(name))
        return self._state.get(name)

    def _resolve(self, chain: tuple[str, ...]):
        ends = [name for name in self._names if name[-len(chain) :] == chain]
        if len(ends) > 1:
            ends = [name for name in ends if name == chain] or ends
        if len(ends) > 1:
            listed = ", ".join("/".join(name) for name in ends)
            raise ValueError(f"State field {'.'.join(chain)!r} is ambiguous; it ends {listed}")
        if ends:
            return self._read(ends[0])
        if any(_contains(name, chain) for name in self._names):
            return self._child(chain)
        raise KeyError(f"State has no field {'.'.join(chain)!r}")

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._resolve((*self._chain, name))
        except KeyError as error:
            raise AttributeError(error.args[0]) from None

    def __getitem__(self, key: str):
        return self._resolve((*self._chain, *key.replace("/", ".").split(".")))

    def __setattr__(self, name: str, value) -> None:
        raise AttributeError("Fields is read-only; write a field with snv.set_field")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Fields is read-only; write a field with snv.set_field")

    def __dir__(self) -> list[str]:
        n = len(self._chain)
        return sorted(
            {name[i + n] for name in self._names for i in range(len(name) - n) if name[i : i + n] == self._chain}
        )

    def __repr__(self) -> str:
        return f"Fields({'.'.join(self._chain) or '<root>'})"
