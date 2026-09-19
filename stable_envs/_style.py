"""Style as a tensor: the appearance half of the state, batched and storable.

The repo's claim is that a frame is a function of a compact state. Appearance is part of that frame, so
if a dataset stores only the dynamic state it can only regenerate the frame under whatever variation
happens to be set at render time — which is not reproduction, it is a coincidence. Style therefore has
to be state too.

``StyleTensor`` is the batched form of a ``spaces.Dict``: one flat ``(B, style_dim)`` float32 tensor
plus the layout that says which columns are ``block.color``. Envs index it by path and get a
``(B, width)`` slice, so the same code serves one shared appearance and 4096 different ones — the
per-environment case is the default rather than a special path, which is what made the earlier
implementation quietly collapse a batch to environment 0's colours.

Storing it is then trivial: a dataset row is ``(state, style)``, both flat float32, and
``render(state, style=style)`` reproduces the frame exactly.

THE BATCH RULE, and why it lives here and nowhere else
------------------------------------------------------

"A per-environment appearance silently became environment 0's" has now been a shipped defect three
times, in three different places: PushT's ``_var`` doing ``.reshape(-1)[:1].expand(B)``, the
static-prefix bake taking ``scene.kind[:1]``, and ``StableEnv.resolve_style`` falling back to
``self._style.values[:1]`` whenever the render batch differed from the env's. Each one passed every
conformance check of its day, because those checks compare an env against itself under one appearance.
The common cause is not carelessness — it is that the operation "make this style fit that batch" was
open-coded at each site, and the wrong way to do it (take row 0, broadcast) is one slice shorter than
the right way.

So there is now exactly ONE operation that changes a style's batch, :meth:`StyleTensor.for_batch`, and
it implements the only rule that is ever correct:

  * the batch already matches                      -> the style is used unchanged;
  * the style is ONE row, or every row is equal    -> it broadcasts, because row *i* IS row 0;
  * anything else                                  -> ``ValueError`` naming both batches.

The third case is not conservatism. ``env.render(states[3:4])`` hands the renderer a ``(1, state_dim)``
array; nothing in it says "these are rows 3..4 of the env's batch", so there is no way to pick the
right appearance and every choice except refusing is a guess. The caller knows the indices and passes
them: ``env.render(states[3:4], style=env.get_style()[3:4])``.

No ``[:1]`` remains on any style path. A collapse cannot be reintroduced by editing an env, only by
editing this rule.
"""

from __future__ import annotations

import numpy as np
import torch


def rows_are_equal(*tensors: torch.Tensor) -> bool:
    """True when every one of ``tensors`` holds the same value in every row of its leading dim.

    The single implementation of "is this batch-uniform?", used by :attr:`StyleTensor.shared` and by
    ``SceneBuilder`` to decide whether a static prefix may be baked. Both used to answer it their own
    way and one of them answered it wrong: the builder tested ``stride(0) != 0``, which is a test for
    *how a tensor was made*, not for what it holds — a ``(B, w)`` slice of a real style tensor has a
    non-zero row stride whether its rows differ or not, so the answer was "per-environment" for every
    batched scene ever built and the bake it guards never fired above ``B == 1``.

    The comparisons are issued together and reduced on device so the whole question costs ONE
    host-device synchronisation regardless of how many tensors are asked about; the tensors this is
    used on are per-primitive parameters (a few hundred kB at ``B = 16,384``), not pixels.
    """
    ok = None
    for t in tensors:
        if t is None or t.ndim == 0 or t.shape[0] <= 1 or t.numel() == 0:
            continue
        eq = (t == t[:1]).all()
        ok = eq if ok is None else (ok & eq)
    return True if ok is None else bool(ok)


def _as_tensor(v, device, dtype=torch.float32) -> torch.Tensor:
    """Accept a torch tensor (on any device), a numpy array or a python scalar/sequence.

    ``np.asarray`` on a CUDA tensor raises, so routing everything through numpy breaks the common case
    of passing values that are already on the GPU — which is what a caller does when the style came
    from a dataset batch or from a previous render.
    """
    if torch.is_tensor(v):
        return v.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(v), device=device, dtype=dtype)


class StyleTensor:
    """``(B, style_dim)`` appearance values, addressable by variation path."""

    def __init__(self, values: torch.Tensor, layout: list[tuple[str, int, int]], shared: bool | None = None):
        self.values = values
        self.layout = {p: (o, w) for p, o, w in layout}
        self.paths = [p for p, _, _ in layout]
        #: True/False when sharedness is known by construction, None when it has to be measured.
        #: Kept lazy on purpose: the answer needs a device-to-host read, and the render hot path never
        #: asks for it (the batch already matches, so ``for_batch`` returns ``self`` untouched).
        self._shared = shared

    @property
    def batch(self) -> int:
        return int(self.values.shape[0])

    @property
    def shared(self) -> bool:
        """True when every environment provably draws the SAME appearance.

        Measured once and cached, never assumed. A style built from a single row (the default
        appearance, ``set_style(one_row)``, an env with no variation at all) knows the answer without
        looking; a style handed over as ``(B, style_dim)`` does not, and the comparison is one small
        kernel plus one sync on a tensor that is a few hundred bytes wide.

        This is what keeps the ergonomic case ergonomic. ``set_style(None)`` fills B identical rows,
        and rendering a sub-batch under it is not a collapse — row 3 IS row 0 — so it must keep
        working, and it does, without the caller having to know that the env stores B rows internally.

        It is also the precondition for the static-prefix bake: a prefix built from a shared
        appearance is batch-uniform, and one built from a per-environment appearance is not. Passing
        this object to ``SceneBuilder(style=...)`` therefore lets the builder answer that question for
        free instead of re-measuring the primitives it just assembled.
        """
        if self._shared is None:
            self._shared = rows_are_equal(self.values)
        return self._shared

    def __contains__(self, path: str) -> bool:
        return path in self.layout

    def __getitem__(self, path: str) -> torch.Tensor:
        """``(B, width)`` for a multi-value leaf, ``(B,)`` for a scalar one."""
        if path not in self.layout:
            raise KeyError(f"unknown style path {path!r}; known: {self.paths}")
        o, w = self.layout[path]
        v = self.values[:, o : o + w]
        return v if w > 1 else v.reshape(-1)

    def get(self, path: str, default=None):
        return self[path] if path in self.layout else default

    def _layout_list(self) -> list[tuple[str, int, int]]:
        return [(p, o, w) for p, (o, w) in self.layout.items()]

    def to(self, device=None, dtype=None) -> StyleTensor:
        return StyleTensor(
            self.values.to(device=device, dtype=dtype or self.values.dtype), self._layout_list(), shared=self._shared
        )

    def for_batch(self, batch: int, owner: str = "render") -> StyleTensor:
        """This style at ``batch`` rows. THE only operation that changes a style's batch.

        See the module docstring for why there is exactly one of these and what it refuses. The refusal
        is the whole point: broadcasting a per-environment style onto a differently-sized batch has no
        correct answer, and the answer it used to give — environment 0's appearance for every row — was
        wrong by up to 229/255 and silent.
        """
        b = int(batch)
        if self.batch == b:
            return self
        if not self.shared:
            raise ValueError(
                f"{owner}: the env carries a PER-ENVIRONMENT style of {self.batch} rows and this call "
                f"asks for {b} environment(s). A style with one row (or with every row equal) "
                f"broadcasts to any batch; {self.batch} different appearances cannot be spread over "
                f"{b} environments, because nothing in the states says which of the {self.batch} rows "
                f"they are. This used to substitute row 0's appearance for every row and say nothing. "
                f"Index the style the same way as the states, e.g. "
                f"env.render(states[i:j], style=env.get_style()[i:j])."
            )
        return StyleTensor(
            self.values[:1].expand(b, self.values.shape[1]).contiguous(), self._layout_list(), shared=True
        )

    @classmethod
    def from_space(cls, space, batch: int, device="cuda", overrides: dict | None = None, values=None) -> StyleTensor:
        """Build from a variation space, optionally overriding leaves per environment.

        ``values`` is an existing ``(B, style_dim)`` tensor (e.g. read from a dataset), and must hold
        either one row (shared by the batch) or exactly ``batch`` rows. Any other row count is refused
        rather than reshaped: the old code let it through, and a caller who sliced their states but not
        their style got a style tensor of the wrong height that either exploded three frames later in
        the scene builder or, worse, broadcast.

        ``overrides`` is ``{path: value}`` where the value is either shared (``w`` numbers) or one row
        per environment (``batch * w`` numbers); it is applied on top, which is what lets a caller
        randomise two knobs and leave the rest at the official defaults.
        """
        layout = space.flat_layout()
        dim = space.flat_dim
        dev = torch.device(device)
        b = int(batch)
        shared: bool | None
        if values is None:
            base = torch.as_tensor(space.pack(), dtype=torch.float32, device=dev)
            vals = base.reshape(1, dim).repeat(b, 1)
            shared = True
        elif dim == 0:
            # Nothing to carry, and ``reshape(-1, 0)`` is an error rather than a no-op. An env with no
            # variation still gets a well-formed style tensor so every path below is uniform.
            vals = torch.zeros((b, 0), dtype=torch.float32, device=dev)
            shared = True
        else:
            v = _as_tensor(values, dev)
            # SHAPE, not size. ``reshape(-1, dim)`` on a (dim, B) array succeeds whenever B == dim and
            # hands back a transposed style whose every column is somebody else's knob — the same
            # silent-reinterpretation the state path had, on the quantity that has already collapsed
            # three times in this repo. Divisibility is never consulted for a multi-axis input.
            if v.ndim >= 2 and v.shape[-1] != dim:
                raise ValueError(
                    f"style rows are {dim} number(s) wide and the last axis of this one is "
                    f"{v.shape[-1]} (shape {tuple(v.shape)}). A style is (batch, style_dim) or one "
                    f"shared (style_dim,) row; anything else used to be flattened and regrouped, "
                    f"which for a transposed array succeeds and gives every environment a mixture of "
                    f"other environments' knobs. env.style_names lists the {dim} columns in order."
                )
            if v.ndim <= 1 and v.numel() % dim:
                raise ValueError(
                    f"a flat style must be a whole number of {dim}-wide rows, and {v.numel()} is not "
                    f"a multiple of {dim}. One row is env.style_dim numbers, in env.style_names order."
                )
            vals = v.reshape(-1, dim)
            shared = True if vals.shape[0] == 1 else None
            if vals.shape[0] == 1 and b != 1:
                vals = vals.repeat(b, 1)
            elif vals.shape[0] != b:
                raise ValueError(
                    f"style has {vals.shape[0]} rows but {b} environment(s) are being rendered. A "
                    f"style must be one row (shared by the whole batch) or exactly one row per "
                    f"environment, in the same order as the states — anything else has no defined "
                    f"meaning, and the row count is exactly what a caller gets wrong when they slice "
                    f"the states and forget the style."
                )
        if overrides:
            lut = {p: (o, w) for p, o, w in layout}
            for path, v in overrides.items():
                if path not in lut:
                    raise KeyError(f"unknown style path {path!r}; known: {sorted(lut)}")
                o, w = lut[path]
                t = _as_tensor(v, dev)
                n = t.numel()
                if t.ndim >= 2 and t.shape[-1] != w and n == b * w:
                    raise ValueError(
                        f"variation {path!r}: expected one value per environment as ({b}, {w}) and "
                        f"got {tuple(t.shape)}, which holds the right NUMBER of values in the wrong "
                        f"arrangement — almost always a transpose. Reshaping it by size would give "
                        f"environment i a mixture of the other environments' values and render "
                        f"without complaint, so it is refused. Pass ({b}, {w}), or ({w},) to share "
                        f"one value across the batch."
                    )
                if n == b * w and b != 1:
                    t = t.reshape(b, w)  # one value per environment
                    if shared:
                        shared = None  # may still be uniform; measure it if asked
                elif n == w:
                    t = t.reshape(1, w).expand(b, w)  # one value shared by the batch
                else:
                    raise ValueError(
                        f"variation {path!r}: expected {w} numbers (shared by the batch) or "
                        f"{b * w} ({b} environments x {w}), got {n} with shape "
                        f"{tuple(t.shape)}. A count that is neither used to be taken as 'the first "
                        f"{w} numbers, broadcast', i.e. environment 0's value for everybody."
                    )
                vals = vals.clone()
                vals[:, o : o + w] = t
        return cls(vals.contiguous(), layout, shared=shared)
