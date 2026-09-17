"""The contract every stable-envs environment implements, and the reason the repo has one at all.

The thesis is that a frame is a pure function of a compact state, so the interface is built around that
single claim and makes it checkable:

    state_dim / state_names       what the compact state IS, and what each number means
    get_state() -> (n, state_dim) read it out
    set_state(states)             put it back; round-trips exactly, and takes a SHORT last batch
    variation_space               every appearance/layout knob, as samplable Spaces
    scene(states, variation)      (state, variation) -> Scene2D, the whole picture
    render(states)                (B,H,W,3) uint8 on device, batched

``scene`` is the load-bearing method. An env that can produce a ``Scene2D`` from a state has *proved*
the frame is a function of the state, because there is nowhere else for information to come from — no
env instance, no simulator handle, no hidden buffer. Envs that cannot do that (the frame depends on
emulator internals, or on assets too complex to describe) do not belong here, and saying so is a
result rather than a failure.

Physics is deliberately NOT part of this contract. Rendering and simulation are separable, and keeping
them separate is what makes the repo useful:

  * the common case is re-rendering a stored dataset of states — no simulator runs at all;
  * where a batched GPU simulator exists it is offered as ``step``, but the reference CPU simulator
    stays the definition of the dynamics;
  * a GPU physics port and a GPU renderer have completely different fidelity stories. A renderer can be
    bit-exact against the reference. A contact solver cannot: contact is chaotic, so any reimplementation
    diverges, and conflating the two lets a "our env matches" claim rest on the half that does not.

The conformance suite in ``tests/gpu`` is parameterised over the registry, so an env added tomorrow is
covered without editing a test: render parity against the official renderer, get/set round-trip,
batch-invariance, variation coverage, and backend agreement.
"""
from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from . import dynamics as _dynamics
from ._config import deprecated as _deprecated
from ._config import get_config as _get_config
from ._style import StyleTensor
from .spaces import Dict as VariationDict
from .specs import CompositeSpec, Spec

#: ``State`` is a plain ``tensordict.TensorDict`` with batch size ``[num_envs]`` — an alias, not a
#: subclass, so anything that consumes a TensorDict consumes a stable-envs state and there is no
#: private type to learn. The keys are grouped by ENTITY (``state["agent", "position"]``), which is
#: what the environment is; the flat packed row every renderer and every frozen corpus uses is the
#: internal layout and is reached through ``env.pack`` / ``env.unpack``.
State = TensorDict

#: What ``step`` returns: a ``TensorDict`` carrying ``"state"`` and, for a cell that HAS them, a
#: ``"reward"`` ``(num_envs,)`` and a ``"done"`` ``(num_envs,)``. An alias for the same reason
#: :data:`State` is one — there is no private type to learn — and a name because the word was being
#: used in five docstrings while nothing defined it, which is how forty cells end up hand-rolling the
#: same ``TensorDict(...)`` call five slightly different ways. Build one with
#: :meth:`Env._step_result`, never by hand.
StepResult = TensorDict

#: How many offending rows/columns a diagnostic lists before it says "and N more". Long enough to see
#: a pattern (one bad row, or a whole column), short enough not to bury the sentence explaining it.
_MAX_LISTED = 8


def require_contiguous_out(owner: str, out) -> None:
    """``render(out=)`` must be handed a destination the renderer can actually write.

    Every backend in this repo — the fused rasterizer, the tile blitter, the ray-caster, and the
    per-env overrides that copy into ``out.view(...)`` — reaches its destination through a flat view
    and writes ``B*H*W*3`` bytes linearly from that view's own offset. ``Tensor.view`` does NOT check
    contiguity when the requested shape is the one the tensor already has: it hands the strided tensor
    straight back. So a strided destination did not raise and did not produce the frames either — it
    scattered them across the caller's buffer, measured at 255/255 on ``buf[:, 1]`` of a
    ``(B, T, H, W, 3)`` rollout buffer, which is the ordinary shape for the idiom the README
    recommends.

    Refusing is chosen over quietly renting a temporary and copying. ``out=`` exists for exactly one
    reason — the caller owns the allocation — so servicing it with a hidden allocation would make the
    argument a different lie. The caller can always assign instead (``frames[:, t] = env.render(...)``),
    which is one copy, visible, and correct; the message says so.
    """
    if out is None:
        return
    if torch.is_tensor(out):
        ok = out.is_contiguous()
    else:                                            # numpy: reshape() of a strided array COPIES
        flags = getattr(out, "flags", None)
        ok = flags is None or bool(flags["C_CONTIGUOUS"])
    # Frames are uint8. A float destination is not refused by any backend's ``view``/``reshape`` — the
    # shape is right and the write succeeds — so the caller gets a buffer of 0..255 floats where they
    # asked for pixels, or a silently converted copy, depending on the path. Checked here so that the
    # per-env overrides that reach ``out`` through their own ``view`` (the tile families, the
    # ray-caster) get the same contract as the two rasterizer backends, which check it in
    # ``raster2d._destination``.
    dtype = getattr(out, "dtype", None)
    if dtype is not None and dtype not in (torch.uint8, np.dtype(np.uint8)):
        raise ValueError(
            f"{owner}: `out` must be uint8 and this one is {dtype}. A frame is 3 bytes per pixel; a "
            f"float destination accepts the write without complaining and hands back 0..255 floats "
            f"where pixels were asked for.")
    if ok:
        return
    raise ValueError(
        f"{owner}: `out` must be contiguous, and this one is not. The renderer writes the frames "
        f"linearly through a flat view of the destination, and a view of a strided tensor keeps its "
        f"strides, so the frames would land across the neighbouring slices of your buffer instead of "
        f"in it — silently, since nothing about the write is invalid. This is what a time or camera "
        f"axis in front of the batch axis produces: `buf[:, t]` of a (B, T, H, W, 3) buffer is "
        f"strided. Either assign (`buf[:, t] = env.render(states)`), which costs one copy, or give "
        f"the buffer a layout whose frame axis is outermost (`buf[t]` of a (T, B, H, W, 3) buffer is "
        f"contiguous and can be rendered into directly).")


def _guard_render_out(fn):
    """Wrap a ``render`` implementation so the two rules every env shares hold before it runs.

    Installed by ``Env.__init_subclass__`` on every subclass that defines ``render``, which is
    the only chokepoint every env genuinely shares: most envs override ``render`` outright (the tile
    families and the ray-caster never build a Scene2D), so a check placed in the base implementation,
    or in one backend, covers a minority of them. MiniGrid used to be the only family that checked,
    which is precisely the shape of a rule that is enforced per-env.

    Three rules, all for that reason:

      * ``out=`` must be contiguous (``require_contiguous_out``);
      * the states must BE states (``check_states``). All 26 overrides open with
        ``torch.as_tensor(states, ...)`` followed by ``reshape(-1, self.state_dim)``, which is the
        silent-garbage path: a frame stack whose size happens to divide by ``state_dim`` reshapes
        without complaint and renders a picture built from pixel bytes read as coordinates. Checking
        here closes it for every env at once, including the ones added tomorrow;
      * ``render()`` with no states draws the states ``set_state`` actually wrote, which after a SHORT
        batch is fewer than ``B``. Those same overrides open with ``s = self._state if states is
        None``, and ``_state`` is the full ``(B, state_dim)`` buffer whose tail is stale — so a short
        batch would render ``B`` frames, the last few of the previous batch. Substituting
        ``get_state()`` here fixes every env at once and keeps ``_state`` at a fixed address, which the
        CUDA-graph capture in ``envs/_stepgraph.py`` requires: rebinding it would leave a replayed
        graph pointing at dead memory. Editing 26 files instead would also leave env 27 wrong.
    """
    @functools.wraps(fn)
    def render(self, *args, **kw):
        out = kw.get("out", args[5] if len(args) > 5 else None)
        if out is not None:
            require_contiguous_out(f"{type(self).__name__}.render", out)
        if args:
            args = (render_states_arg(self, args[0]), *args[1:])
        else:
            kw["states"] = render_states_arg(self, kw.get("states"))
        return fn(self, *args, **kw)
    render._se_out_guarded = True
    return render


def _as_state_tensor(owner: str, states) -> torch.Tensor:
    """Anything array-like -> a tensor, WITHOUT casting, so the dtype is still evidence.

    The cast to float32 has to happen after the checks rather than inside the conversion: ``uint8`` is
    the single loudest signal that what arrived is a batch of frames, and converting first erases it.

    A list or tuple of tensors is stacked rather than refused, which is what makes the most common
    mistake in the recommended idiom a non-event: a ``DataLoader`` over ``TensorDataset(states)`` yields
    ``[states]`` — a one-element list — and ``torch.as_tensor`` on it dies with "only one element
    tensors can be converted to Python scalars", which names neither the caller nor the fix.
    """
    if isinstance(states, Mapping):
        raise TypeError(
            f"{owner}: got a {type(states).__name__}, not an array of states. A TrajectoryDataset "
            f"batch is a dict — the states are under the 'states' key, so pass batch['states'] "
            f"(and note it carries a time axis: (B, window, state_dim)).")
    if torch.is_tensor(states):
        return states
    if isinstance(states, (list, tuple)) and len(states) and all(
            torch.is_tensor(x) or isinstance(x, np.ndarray) for x in states):
        parts = [torch.as_tensor(x if torch.is_tensor(x) else np.asarray(x)) for x in states]
        # A ONE-element sequence is unwrapped rather than stacked, because that is what it is: a
        # DataLoader over TensorDataset(states) yields ``[batch]``, and stacking would add a leading
        # axis and then trip the "states are 2-D" rule on the package's own recommended idiom. A
        # longer sequence is a list of samples and stacking is what it means.
        return parts[0] if len(parts) == 1 else torch.stack(parts)
    try:
        return torch.as_tensor(states)
    except Exception as e:                                   # noqa: BLE001 — re-raised with the fix
        raise TypeError(
            f"{owner}: could not read {type(states).__name__} as an array of states ({e}). It wants "
            f"a (batch, state_dim) float array — a torch tensor, a numpy array, or anything "
            f"torch.as_tensor accepts.") from e


def _nonfinite_report(s: torch.Tensor, names: tuple[str, ...]) -> str | None:
    """Which rows and which named columns hold a NaN or an inf, or None if all are finite.

    Run on the tensor the CALLER handed over, before it is copied to the device. That is not an
    optimisation detail so much as the reason the check can be unconditional: in the idiom this repo
    recommends the states arrive from a ``DataLoader`` on the host, so the scan is a CPU reduction over
    a few kilobytes and costs no device synchronisation at all.
    """
    bad = ~torch.isfinite(s)
    if not bool(bad.any()):
        return None
    rows = torch.nonzero(bad.any(1), as_tuple=False).flatten().tolist()
    cols = torch.nonzero(bad.any(0), as_tuple=False).flatten().tolist()
    named = [f"{names[c]} (column {c})" if c < len(names) else f"column {c}" for c in
             cols[:_MAX_LISTED]]
    kinds = []
    if bool(torch.isnan(s).any()):
        kinds.append("NaN")
    if bool(torch.isinf(s).any()):
        kinds.append("inf")
    more_r = f" and {len(rows) - _MAX_LISTED} more" if len(rows) > _MAX_LISTED else ""
    more_c = f" and {len(cols) - _MAX_LISTED} more" if len(cols) > _MAX_LISTED else ""
    return (f"{len(rows)} of {s.shape[0]} state(s) hold {'/'.join(kinds)}: "
            f"row(s) {rows[:_MAX_LISTED]}{more_r}, in {', '.join(named)}{more_c}")


def render_states_arg(env, states) -> torch.Tensor:
    """What ``render`` should actually draw: the live state, or the caller's array once it is checked.

    Split out so the base ``render`` and the wrapper installed on every override run the identical
    rule. Two decisions live here.

    ``states is None`` means "what the env holds", and that is ``get_state()`` rather than ``_state``,
    because after a short ``set_state`` the tail of the buffer is the previous batch.

    The NaN/inf scan is skipped for a tensor that is ALREADY ON THE DEVICE, and that is a measured
    trade rather than an oversight. The scan needs one device-to-host read, and ``render`` is the
    streaming path — ``render_states`` calls it once per chunk, a training loop once per step — so an
    unconditional scan would put a synchronisation in the one place this repo works to keep free of
    them. Every entry point where user data ENTERS from the host still scans unconditionally:
    ``set_state`` (the training-step surface), ``render_states``, and both dataset constructors. A
    device tensor handed straight to ``render`` was produced by the caller's own kernels, one step
    after whichever entry point did check it.
    """
    env._check_open("render")
    if states is None:
        return env.get_state()
    if isinstance(states, TensorDict):
        # A public State is accepted wherever packed rows are (contract §5). ``pack`` is the only
        # bridge between the two and it is the env's own, so an env whose State carries a field the
        # packed row does not (PushT's agent velocity) drops exactly that field here and the renderer
        # sees the identical Step-8 row.
        return env.pack(states)
    scan = not (torch.is_tensor(states) and states.device.type == "cuda")
    return check_states(env, states, f"{type(env).__name__}.render", leading=True, scan=scan)


def _names_phrase(names: tuple[str, ...], limit: int = 8) -> str:
    if not names:
        return ""
    shown = ", ".join(names[:limit]) + (", ..." if len(names) > limit else "")
    return f" ({shown})"


def check_states(env, states, owner: str | None = None, leading: bool = False,
                 scan: bool = True, move: bool = True) -> torch.Tensor:
    """``states`` -> the same numbers as float32 on ``env.device``, SHAPE UNCHANGED, or a real message.

    THE SILENT-GARBAGE PATH THIS CLOSES. Every entry point that turns numbers into pixels used to go
    through ``torch.as_tensor(states, ...).reshape(-1, state_dim)``, and that reshape is the most
    dangerous line in the package, because two of its three failure modes produce a *valid* array:

      * **frames handed in where states were meant.** A PushT frame is 96*96*3 = 27,648 numbers and
        ``state_dim`` is 5; 27,648 is divisible by 5 only for some shapes, but whether it divides is
        arithmetic, not intent, and sooner or later it divides. When it does, the reshape succeeds and
        the env renders a plausible-looking picture built out of pixel bytes reinterpreted as
        coordinates. No exception, no warning, a frame that looks like a frame.
      * **the wrong state width.** A ``(B, 7)`` array for a 5-wide env reshapes into 1.4 B states that
        each straddle two of the caller's rows, and the frames are again plausible.
      * **NaN or inf.** The rasterizer clamps and draws somewhere, or draws nothing; either way the
        frame is a frame and the loss curve is a loss curve. Same defect wearing different clothes.

    So the rule is now shape-based rather than size-based, which is what makes it impossible rather
    than unlikely: **a state array is 2-D and its last axis is exactly ``state_dim``.** Divisibility is
    never consulted for a multi-axis input, so ``(B, H, W, 3)`` is refused whether or not it happens to
    divide. ``leading=True`` relaxes only the rank, for the two entry points whose documented contract
    is "arbitrary leading dims" (``Env.render``, ``render_states``) — the last axis is still
    required to be exactly ``state_dim`` there, which is the half that was load-bearing.

    ``scan`` controls the NaN/inf scan. It runs on the tensor the caller handed over, before the copy
    to the device, so for host-side input — the recommended idiom, where states come off a
    ``DataLoader`` — it is a CPU reduction over a few kilobytes and costs no device synchronisation.
    ``Env.render`` turns it off for a tensor already on the device, and says why.

    ``move=False`` converts the dtype but leaves the tensor where it is, for ``render_states``, whose
    whole contract is that a corpus larger than device memory passes through a chunk at a time.

    The batch is deliberately NOT checked here: ``render`` takes any batch, and ``set_state`` has its
    own rule (shorter than the env is fine, longer is not). The caller decides.
    """
    owner = owner or f"{type(env).__name__}.set_state"
    D = int(env.state_dim)
    names = tuple(getattr(env, "state_names", ()) or ())
    raw = _as_state_tensor(owner, states)
    shape = tuple(raw.shape)
    dt = str(raw.dtype).replace("torch.", "")

    if raw.ndim == 0:
        raise ValueError(f"{owner}: got a scalar. A state is {D} number(s)"
                         f"{_names_phrase(names)}, and a batch of them is (batch, {D}).")

    # A frame stack, named as one. Checked before the rank rule so the message is about what arrived
    # rather than about how many axes it has. Two independent signals, because neither alone covers
    # it: a channel-shaped last axis that is NOT the state width, and uint8 at rank 3 or more (states
    # are float, and for a 3-wide env a (H, W, 3) frame is indistinguishable from a corpus by shape).
    if raw.ndim >= 3 and (raw.dtype == torch.uint8
                          or (shape[-1] in (1, 3, 4) and shape[-1] != D)):
        cast = (" If these really are states stored as uint8, say so: states.astype('float32')."
                if raw.dtype == torch.uint8 else "")
        raise ValueError(
            f"{owner}: expected states shaped (batch, {D}) and got {shape} {dt}. That is a batch of "
            f"FRAMES, not states"
            + (", and uint8 says so — states are float" if raw.dtype == torch.uint8 else "") +
            f". set_state/render run in the other direction: they take the {D} number(s) a frame is "
            f"drawn FROM{_names_phrase(names, 6)} and produce the pixels — "
            f"`frames = env.set_state(states).render()`. This used to be reshaped to (-1, {D}) and "
            f"rendered whenever the size happened to divide, which is a plausible-looking picture "
            f"built from pixel bytes read as coordinates.{cast}")

    if raw.ndim > 2 and not leading:
        raise ValueError(
            f"{owner}: states are 2-D — (batch, {D}) — and this input has {raw.ndim} axes: {shape}. "
            f"A rank higher than 2 is refused outright rather than flattened, because flattening is "
            f"exactly how a frame stack used to become a batch of states whenever its size happened "
            f"to divide by {D}. If this is a corpus with a time or trajectory axis, say so: "
            f"`states.reshape(-1, {D})` (and build the env at that batch). If it is frames, you want "
            f"env.render, not env.set_state.")

    if raw.ndim >= 2 and shape[-1] != D:
        raise ValueError(
            f"{owner}: {type(env).__name__} expects state_dim {D}, got {shape[-1]} "
            f"(input shape {shape}, dtype {dt}). A state is {D} number(s) wide"
            f"{_names_phrase(names, 12)}. This used to be reshaped to (-1, {D}), which regroups the "
            f"input into {raw.numel() // D if raw.numel() % D == 0 else 'a different number of'} "
            f"states that each straddle two of your rows, and renders without complaint. Check the "
            f"state layout with env.state_names / env.describe().")

    if raw.ndim == 1 and raw.numel() % D:
        raise ValueError(
            f"{owner}: a flat input must be a whole number of {D}-wide states, and {raw.numel()} is "
            f"not a multiple of {D} (input shape {shape}, dtype {dt}). One state is {D} number(s)"
            f"{_names_phrase(names, 12)}; a batch is (batch, {D}).")

    if raw.numel() == 0:
        raise ValueError(
            f"{owner}: no states to render — got 0 of them (input shape {shape}). An empty batch "
            f"produces an empty frame buffer, which is almost never what a training step wants; "
            f"check the sampler, or the slice that produced it.")

    if scan:
        bad = _nonfinite_report(raw.reshape(-1, D), names)
        if bad:
            raise ValueError(
                f"{owner}: {bad}. A non-finite state renders SILENTLY — the rasterizer clamps it and "
                f"draws somewhere plausible, or draws nothing — so the frame looks like data and the "
                f"model trains on it. Check the step that produced these states (a diverged rollout "
                f"and a divide-by-zero in a normalisation both land here).")
    raw = raw.detach() if torch.is_tensor(raw) else raw
    return raw.to(device=env.device, dtype=torch.float32) if move else raw.to(dtype=torch.float32)


#: The fidelity tiers an env may declare. THE TIER IS PART OF THE CONTRACT, not a caveat in a README.
#:
#: ``"parity"``  the frame is the OFFICIAL frame. Every interior pixel is bit-identical to the
#:               reference renderer (or as close as ``parity_interior_min`` declares, with a stated
#:               mechanism). A dataset regenerated from states is interchangeable with a stored one:
#:               a policy trained on regenerated pixels can be evaluated on official pixels, because
#:               they are the same pixels.
#:
#: ``"visual"``  the frame is a DIFFERENT RENDER of the same scene: same geometry, same camera, same
#:               materials, drawn by a different rasterizer that is much faster and does not agree
#:               pixel for pixel. Suitable for generating training data live. NOT a substitute for a
#:               stored dataset when a policy will be evaluated on official pixels — the train/eval
#:               pixel distributions differ, by an amount the env declares in ``visual_fidelity``, and
#:               nothing in this repo measures how much that costs a policy.
#:
#: The two tiers exist because the second use is real and the first bar is the wrong one for it. They
#: are kept apart mechanically rather than by convention: a visual-tier env sets
#: ``parity_interior_min = None``, so the parity assertion cannot be applied to it, cannot be silently
#: relaxed for it, and cannot be passed by it. See ``tests/gpu/test_conformance.py``.
FIDELITY_TIERS = ("parity", "visual")

#: WHAT DRAWS THE FRAME. A second axis, orthogonal to the tier, and it exists because the tier alone
#: stopped being able to answer the question a consumer actually asks.
#:
#: ``"renderer"``  a rasterizer or ray-caster whose OUTPUT IS THE FRAME. Every env in this repo was
#:                 this until 2026-08-11, which is why it was never worth naming.
#:
#: ``"learned"``   the renderer's output was then passed through a FIXED NEURAL CHECKPOINT trained to
#:                 predict the official frame. The frame is still a pure deterministic function of
#:                 ``(state, style)`` — the checkpoint is fixed, the render feeding it is already such
#:                 a function, and ``render`` is still ``f(get_state())``, still batch-invariant,
#:                 still bit-reproducible by anyone holding the weights. THE CONTRACT IS UNCHANGED;
#:                 WHAT CHANGED IS THE DECODER. The tier is still ``"visual"`` and the declared error
#:                 is still measured against the official renderer, but the residual is no longer "a
#:                 different rasterizer disagrees with this one": part of the picture is SYNTHESISED,
#:                 with the right statistics and the wrong pixels, and a frozen pretrained encoder can
#:                 tell the difference even where the pixel metrics improve.
#:
#: The reason this is a declared field and not a footnote: someone who intends to evaluate a policy on
#: official pixels needs to know which of the two produced their training frames, and THE ANSWER IS
#: NOT VISIBLE IN THE PICTURE. It is visible here, in ``describe()``, in ``format_registry()`` and in
#: the class docstring, which ``check_fidelity_declaration`` requires.
RENDER_DECODERS = ("renderer", "learned")

#: What a ``"visual"`` env must put in ``visual_fidelity``. Three MEASURED numbers, three THRESHOLDS
#: the conformance suite asserts, and the conditions the measurement was taken under. The thresholds
#: are separate from the measurements on purpose: a threshold equal to the measurement flakes on the
#: next state draw, and a measurement inferred from a threshold is not a measurement.
REQUIRED_VISUAL_FIDELITY = ("mean_abs", "within8_frac", "ssim",
                            "mean_abs_max", "within8_min", "ssim_min", "measured_at")

#: The reference package versions a declaration's numbers were measured AGAINST, as a field rather
#: than as prose inside ``measured_at``. Optional, because most declarations predate it; validated
#: whenever present; and read by the certifier, which compares it with the versions the measuring
#: interpreter recorded and marks a mismatch ``blocked_environment`` rather than ``fail`` -- the
#: same rule that already applies to a missing reference package. It exists because of a measured
#: case: DM Control's trackcom cells were declared under mujoco 3.11.0 and judged under 3.12.0, and
#: 3.12.0 re-anchors finite-plane textures (changelog 3.12.0, commit cc7fb98c), so the floor checker
#: the reference drew had moved while ours had not. Judged as a FAIL, that read as a renderer defect;
#: it was a version drift, and a field the certifier can compare is what tells the two apart.
#:
#: Keys follow ``examples/fidelity_step4.py``'s ``reference_packages`` record -- ``"mujoco"``,
#: ``"dm_control"``, ``"gymnasium"`` ... -- so the comparison is a dictionary lookup and not a parse.
#: Values are the exact version strings. A visual env carries it inside ``visual_fidelity`` (next to
#: ``measured_at``, whose text it must agree with); a parity env carries it as the class attribute
#: ``reference_versions``. ``check_fidelity_declaration`` validates the shape; ``reference_versions_of``
#: reads it for either tier, ``visual_fidelity`` taking precedence over the class attribute.
#:
#: REFERENCE-SIDE PACKAGES ONLY. The field gates on a mismatch, so it must name what the OFFICIAL frame
#: depends on and nothing that OUR frame depends on: with ``mujoco_warp`` in it, bumping our own
#: renderer would turn a real regression of ours into ``blocked_environment``. The packages our side
#: was drawn with go in ``MEASURED_WITH_KEY``, which is recorded and never compared.
REFERENCE_VERSIONS_KEY = "reference_versions"

#: The versions OUR renderer's dependencies had when the declaration was measured (``mujoco_warp``,
#: ``warp``, ``nvdiffrast`` ...). Provenance, not a gate: the certifier records it and never blocks on
#: it, because a change on our side is exactly what a fidelity check exists to catch. Same shape as
#: ``REFERENCE_VERSIONS_KEY``; lives in ``visual_fidelity`` beside it.
MEASURED_WITH_KEY = "measured_with"


def reference_versions_of(cls) -> dict[str, str]:
    """The declared reference versions for ``cls``, whichever tier carries them. ``{}`` if none."""
    out = dict(getattr(cls, REFERENCE_VERSIONS_KEY, {}) or {})
    out.update((getattr(cls, "visual_fidelity", {}) or {}).get(REFERENCE_VERSIONS_KEY) or {})
    return out


def _check_reference_versions_shape(cls, where: str, rv) -> None:
    if rv is None:
        return
    if not isinstance(rv, dict) or not rv:
        raise ValueError(
            f"{cls.__name__}: {where} must be a non-empty dict of package -> exact version string, "
            f"got {rv!r}. An empty or non-dict value declares nothing; leave the field out instead.")
    for k, v in rv.items():
        if not (isinstance(k, str) and k and isinstance(v, str) and v.strip()):
            raise ValueError(
                f"{cls.__name__}: {where}[{k!r}] = {v!r}; every entry is a package name mapped to the "
                f"exact version string the declaration was measured against (e.g. 'mujoco': "
                f"'3.11.0'), so the certifier can compare it with what the measuring interpreter "
                f"recorded.")


class Env:
    """Base class: a batched, GPU-renderable environment defined by ``scene()``.

    Step 9 renamed this from ``StableEnv`` and gave it the lifecycle every ``stable-*`` package has —
    ``reset``/``step``/``transition``/``unroll``/``render``/``close``, a TensorDict :data:`State`, and
    the four specs — WITHOUT changing what any of it computes. The packed float32 row
    (``state_dim``/``state_names``) is still the internal layout every renderer, every frozen corpus
    and every certification check uses, and ``pack``/``unpack`` are the only bridge between it and the
    public entity-grouped state. ``StableEnv`` remains as a deprecated alias for this class — the same
    class object, so ``isinstance``, ``issubclass`` and every ``__init_subclass__`` hook are unaffected.
    """

    #: dataset name, hyphenated-lowercase; the env is identified by its class name
    name: str = ""
    #: width of the compact state vector
    state_dim: int = 0
    #: one label per state dimension — printed in errors and used by the round-trip test
    state_names: tuple[str, ...] = ()
    #: native output resolution of the reference renderer, for parity tests
    default_size: int = 96
    #: the reference (CPU, official) env id this is validated against; None means no external reference
    reference_id: str | None = None
    #: Required fraction of INTERIOR pixels that must be bit-identical to the official renderer.
    #: 1.0 (exact) is the default and the expectation: an interior disagreement is a real geometry or
    #: colour error. An env may relax it ONLY with ``parity_note`` explaining the mechanism, which the
    #: test prints. This exists so a genuine, understood limitation is declared in the env and visible
    #: in the report, instead of being hidden by loosening the shared threshold for everyone — the
    #: motivating case is MuJoCo's Gouraud shading of tessellated spheres and capsules, which no
    #: per-pixel analytic shader reproduces exactly.
    parity_interior_min: float | None = 1.0
    #: Required fraction of ALL pixels that must be bit-identical, when the shared bar in
    #: ``tests/gpu/test_conformance.py`` (0.90) is the wrong one for this env. ``None`` means "use the
    #: shared bar", which is what every env should do unless it can say why it cannot.
    #:
    #: It exists because all-pixel exactness is not a property of a renderer alone — it is that
    #: property WEIGHTED BY HOW MUCH OF THE FRAME MOVES. Two ports can share a backend, a shading rule
    #: and a residual per mover pixel and still land on different all-pixel numbers because one scene's
    #: movers cover 9% of the frame and another's cover 13%. The interior floor is the number that
    #: describes the RENDERER (interior pixels are where a disagreement is a real geometry or colour
    #: error) and it is not relaxed by this; this only stops a scene being failed for the size of its
    #: robot. Relaxing it requires ``parity_note``, exactly as ``parity_interior_min`` does.
    parity_all_exact_min: float | None = None
    #: Why ``parity_interior_min`` or ``parity_all_exact_min`` is below its bar, and what would close
    #: it. Required if either is.
    parity_note: str = ""
    #: ``"parity"`` or ``"visual"`` — see ``FIDELITY_TIERS``. Defaults to the strong claim, so a tier
    #: is something an env OPTS OUT of explicitly and visibly, never something it drifts into.
    fidelity_tier: str = "parity"
    #: MEASURED visual agreement with the official renderer, plus the thresholds the suite asserts.
    #: Required, and required to be non-empty, exactly when ``fidelity_tier == "visual"``; forbidden
    #: otherwise, so the two tiers cannot be blurred by an env that declares both.
    visual_fidelity: dict[str, Any] = {}
    #: What the residual IS, where it comes from, and what the tier does and does not license.
    visual_note: str = ""
    #: ``"renderer"`` or ``"learned"`` — see ``RENDER_DECODERS``. Defaults to the plain answer, so a
    #: learned decoder is something an env OPTS INTO explicitly and visibly, never something a change
    #: of render path slides it into.
    render_decoder: str = "renderer"
    #: ``visual_fidelity``, measured again with a per-state appearance DRAWN FROM ``variation_space``
    #: instead of at the default. Optional for a renderer and REQUIRED for a learned decoder.
    #:
    #: The asymmetry is measured, not stylistic. A rasterizer that agrees with the official one at one
    #: camera agrees at another, because it is the same arithmetic pointed somewhere else — so one
    #: declaration at the default appearance describes the whole space. A learned decoder does not
    #: generalise that way and the size of the gap is the whole story: a corrector fitted at one camera
    #: pose scores 3.25/255 there and 9.06 under the same env's declared ±8° of pan, which is worse
    #: than not correcting at all (8.91) and worse than the render it replaces (7.93). A
    #: default-appearance declaration would have said 3.25, and every consumer using the variation
    #: knobs — which is what this family exists for — would have got the worst of the three.
    variation_fidelity: dict[str, Any] = {}
    #: For a PARITY env: the reference package versions its parity numbers were measured against
    #: (a visual env puts the same dict inside ``visual_fidelity``). See ``REFERENCE_VERSIONS_KEY``.
    reference_versions: dict[str, str] = {}

    #: How the packed float32 row maps onto the public, entity-grouped :data:`State`:
    #: ``((key path, width, offset in the packed row or None), ...)``.
    #:
    #: ``None`` for an offset means the field is REAL DYNAMICAL STATE THAT THE PACKED ROW DOES NOT
    #: CARRY — PushT's agent velocity is the motivating case. Such a field is part of the public state
    #: (so ``transition`` can be pure), is filled with zeros by ``unpack`` (which is exactly the
    #: condition ``set_state`` puts the env in, and the reference's condition after ``reset``), and is
    #: dropped by ``pack``, so the packed row, its digests and every frozen corpus are untouched.
    #:
    #: An entry may carry a FOURTH element, the public dtype of that field (``torch.float64`` for a
    #: cell whose claim is near-bit-exactness against a float64 reference); ``unpack`` and ``state``
    #: honour it and ``pack`` still writes the float32 row, so digests and corpora are untouched.
    #:
    #: Left empty, an env gets the trivial layout ``{"state": (state_dim,)}``: no ceremony for a cell
    #: whose state is one vector (contract §3).
    state_layout: tuple[tuple, ...] = ()

    def __init__(self, num_envs: int = 1, size: int | None = None, supersample: int = 1,
                 device: str | None = None, backend: str = "auto", seed: int = 0,
                 batch: int | None = None):
        """``num_envs`` environments on ``device``. ``batch=`` is the deprecated Step-8 spelling.

        The positional order is unchanged from Step 8 (``num_envs`` sits where ``batch`` sat), so the
        existing call sites keep working while Workstream K migrates them.
        """
        if batch is not None:
            _deprecated("Env(batch=)",
                        "stable-envs: `batch=` is deprecated and will be removed; it is now "
                        "`num_envs=` (env = snv.PushT(num_envs=64, device='cuda')). The two mean "
                        "exactly the same thing.")
            if num_envs not in (1, batch) :
                raise TypeError(
                    f"{type(self).__name__}: given both num_envs={num_envs} and batch={batch}. They "
                    f"are the same argument; pass num_envs alone.")
            num_envs = batch
        if device is None:
            device = _get_config()["default_device"] or "cuda"
        self.B = int(num_envs)
        self.size = int(size or self.default_size)
        self.ss = int(supersample)
        self.device = torch.device(device)
        self.backend = backend
        self.rng = np.random.default_rng(seed)
        self.variation_space: VariationDict = self.make_variation_space()
        self._state = torch.zeros(self.B, self.state_dim, device=self.device, dtype=torch.float32)
        #: How many rows of ``_state`` the last ``set_state`` actually wrote. Equal to ``B`` except
        #: after a SHORT batch, which is the ordinary last batch of any ``DataLoader`` built with
        #: ``drop_last=False``; see ``set_state``.
        self._live_rows = self.B
        # Style is state. A dataset that stores only the dynamic half can regenerate a frame only under
        # whatever appearance happens to be set at render time, which is a coincidence rather than
        # reproduction — so the appearance gets its own flat, per-environment, storable tensor.
        self._style = StyleTensor.from_space(self.variation_space, self.B, device=self.device)
        #: Resources are released explicitly (``close``/``with``), never in ``__del__`` — plan §2. A
        #: closed env refuses to render rather than handing back frames drawn through a backend whose
        #: context may already be gone.
        self._closed = False
        self._seed = int(seed)

    def __init_subclass__(cls, **kw):
        """Give every env the shared ``out=`` contract, whether or not its author remembered it.

        A subclass that overrides ``render`` gets its override wrapped once, at class-creation time;
        a subclass that inherits one inherits the already-wrapped function and is not double-wrapped.
        The alternative — asking each env to call a helper — is what produced the defect this exists
        to close, so it is not offered.
        """
        super().__init_subclass__(**kw)
        fn = cls.__dict__.get("render")
        if fn is not None and not getattr(fn, "_se_out_guarded", False):
            cls.render = _guard_render_out(fn)

    # to implement

    def make_variation_space(self) -> VariationDict:
        """Every appearance/layout knob this env exposes. Override; may be an empty Dict."""
        raise NotImplementedError

    def scene(self, states: torch.Tensor, variation: dict | None = None, style=None) -> Scene2D:
        """``(B,state_dim)`` -> ``Scene2D``. The whole picture, and the proof of the thesis.

        Optional for envs that are not drawn from vector primitives. A tile-based env (MiniGrid and its
        relatives) is rendered by gathering from a precomputed atlas, which is exact by construction
        rather than by matching a rasterizer, and forcing it through a polygon IR would only re-derive
        images upstream already caches. Such an env overrides ``render`` instead and leaves this
        unimplemented; the conformance suite skips the checks that need a Scene2D and applies the rest.

        An env whose world rectangle is measured in OUTPUT PIXELS rather than in world units — the
        highway family, whose ``scaling`` is fixed so a bigger frame shows more road — declares an
        extra ``size`` keyword here, meaning "the output size this scene will be rasterized at". That
        is how a caller can ask whether ``size`` is a resolution or a field of view without rendering
        anything and without the env remembering the question: build the scene at S and at 2S and
        compare ``world``. Envs that ignore resolution do not declare the keyword, and its absence is
        itself the answer.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not build a Scene2D; it must override render() instead")

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """``(batch, state_dim)`` states from the env's own reset distribution.

        Used by the conformance tests to exercise the renderer over the region of state space the env
        actually visits — a parity test on states the env never reaches proves very little.
        """
        raise NotImplementedError

    # provided

    def get_state(self) -> torch.Tensor:
        """``(n, state_dim)`` — the states currently held, where ``n`` is what ``set_state`` wrote.

        ``n`` is ``B`` unless the last ``set_state`` was given a short batch, in which case the rows
        beyond it are whatever was there before and are deliberately not handed back.
        """
        return self._state if self._live_rows == self.B else self._state[:self._live_rows]

    def set_state(self, states, style=None) -> "Env":
        """``(n, state_dim)`` states, ``n <= B``. Returns ``self``, so ``.render()`` chains off it.

        A batch SHORTER than the env's is supported rather than refused, and that is the important
        case: every ``DataLoader`` built with ``drop_last=False`` — the default — ends its epoch with a
        short batch, so refusing one would make the ordinary loop either crash on its last step or
        throw data away. Only those ``n`` rows are stored, ``get_state`` returns only those rows, and
        ``render()`` draws only those frames. It costs nothing: measured on one H200, PushT at 96 px
        renders 4,095 states in 3.371 ms against 3.341 ms for 4,096, and 1,023 in 2.773 ms against
        2.771 ms for 1,024 — the renderer's work is per state, and no cache or kernel is keyed on the
        batch.

        A batch LONGER than the env's still cannot be serviced: there is nowhere to put the extra rows,
        and the message says how to fix it either way.

        Everything else about the input is checked in ``check_states``, which is where the reasons
        live. In short: frames handed in where states were meant, a wrong state width, and NaN/inf all
        used to reshape into something valid and render a picture that looks like data.
        """
        s = check_states(self, states, f"{type(self).__name__}.set_state").reshape(-1, self.state_dim)
        n = int(s.shape[0])
        if n > self.B:
            raise ValueError(
                f"{type(self).__name__}.set_state: this env holds {self.B} environment(s) and got {n} "
                f"states. A batch may be SHORTER than the env — the short final batch of a DataLoader "
                f"with drop_last=False renders fine, and only those rows are drawn — but it cannot be "
                f"longer, because there is nowhere to put the extra {n - self.B}. Either build the env "
                f"at the batch you feed it, se.make({type(self).__name__!r}, "
                f"batch={n}), or render without touching the env's buffer: env.render(states), which "
                f"takes any batch.")
        self._state[:n].copy_(s)
        self._live_rows = n
        if style is not None:
            self.set_style(style)
        return self

    @torch.no_grad()
    def render(self, states=None, size: int | None = None, supersample: int | None = None,
               variation: dict | None = None, style=None, out=None) -> torch.Tensor:
        """``(...,state_dim)`` -> ``(...,H,W,3)`` uint8 on device. Arbitrary leading dims."""
        require_contiguous_out(f"{type(self).__name__}.render", out)
        # The base implementation is NOT wrapped by __init_subclass__ (that hook only sees subclass
        # definitions), so the shared rule is applied here explicitly and by the same function.
        s = render_states_arg(self, states)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim)
        sc = self.scene(s, variation, style)
        # Imported HERE rather than at module scope, and that is a dependency rule rather than a
        # style choice (contract §8): the core modules — env, dynamics, specs, registry, _config —
        # import nothing from ``render/``, ``envs/`` or ``interop``, so the arrow between the vocabulary
        # of the package and its backends points one way and cannot become a cycle. ``render`` is the
        # one core method that genuinely needs a backend, and it needs it at call time.
        from .render.raster2d import render as render_scene
        img = render_scene(sc, size or self.size, supersample if supersample is not None else self.ss,
                           backend=self.backend, out=out)
        return img.view(*lead, *img.shape[1:]) if lead else img

    # style (the appearance half of the state)

    @property
    def style_dim(self) -> int:
        return self.variation_space.flat_dim

    @property
    def style_names(self) -> list[str]:
        return [p for p, _, _ in self.variation_space.flat_layout()]

    def default_style(self) -> np.ndarray:
        """``(style_dim,)`` — the OFFICIAL appearance. Rendering with it reproduces the reference."""
        return self.variation_space.pack()

    def get_style(self) -> torch.Tensor:
        return self._style.values

    def set_style(self, style=None, **overrides) -> torch.Tensor:
        """``(B, style_dim)`` (or one shared row), plus optional ``path=value`` overrides.

        Built for ``self._live_rows`` rows rather than for ``B`` so that ``set_state(short, style=short)``
        pairs row for row. With the ordinary full batch the two are the same number.
        """
        self._style = StyleTensor.from_space(self.variation_space, self._live_rows, device=self.device,
                                             values=style, overrides=overrides or None)
        return self._style.values

    def sample_style(self, batch: int | None = None, paths=None,
                     rng: np.random.Generator | None = None) -> np.ndarray:
        """``(batch, style_dim)`` independent appearances — batched domain randomisation.

        ``paths`` restricts randomisation to named leaves; everything else stays at the official
        default, which is what makes a study of "vary only the background" expressible.

        A path that is not a leaf of this env's variation space is an ERROR, not a no-op. The loop
        below is a filter, so an unknown or misspelled path used to match nothing and return the
        default appearance for every environment — a domain-randomisation study that silently
        randomised nothing, which looks exactly like a study whose randomisation did not matter.
        """
        rng = rng or self.rng
        B = int(batch or self.B)
        if paths is not None:
            known = [p for p, _, _ in self.variation_space.flat_layout()]
            unknown = [p for p in paths if p not in known]
            if unknown:
                raise KeyError(
                    f"{type(self).__name__}.sample_style: unknown variation path(s) {unknown}. This "
                    f"used to match nothing and hand back the DEFAULT appearance for every "
                    f"environment, which is indistinguishable from a randomisation that had no "
                    f"effect. Known paths: {known}")
        out = np.tile(self.default_style(), (B, 1))
        for path, off, w in self.variation_space.flat_layout():
            if paths is not None and path not in paths:
                continue
            node = self.variation_space
            for part in path.split("."):
                node = node[part]
            if hasattr(node, "low"):
                lo = np.broadcast_to(node.low, node.shape).astype(np.float64).reshape(-1)
                hi = np.broadcast_to(node.high, node.shape).astype(np.float64).reshape(-1)
                out[:, off:off + w] = rng.uniform(lo, hi, size=(B, w))
            elif hasattr(node, "n"):
                out[:, off:off + w] = rng.integers(0, int(node.n), size=(B, w))
        return out.astype(np.float32)

    def resolve_style(self, style=None, variation: dict | None = None, batch: int | None = None
                      ) -> StyleTensor:
        """The style an env's ``scene`` should draw with: explicit tensor, dict overrides, or current.

        Every env in the registry funnels through this, and it is the one place that decides how a
        stored per-environment appearance meets a render of a different size. It decides by asking
        ``StyleTensor.for_batch``, which broadcasts a shared appearance and REFUSES a per-environment
        one — see ``core/style.py`` for why that rule exists and what the three shipped collapses had
        in common. There is deliberately no ``[:1]`` here any more; taking row 0 was the bug, in both
        branches (the ``variation=`` branch collapsed even when the batch matched, so
        ``set_state(st, style=per_env)`` followed by ``render(variation={...})`` drew eight copies of
        environment 0).
        """
        B = int(self.B if batch is None else batch)
        if style is None:
            # only now is the env's own style load-bearing, so only now can its batch be wrong
            current = self._style.for_batch(B, owner=f"{type(self).__name__}.render")
            if not variation:
                return current
            style = current.values
        return StyleTensor.from_space(self.variation_space, B, device=self.device,
                                      values=style, overrides=variation)

    def sample(self, steps: int = 100, batch: int | None = None, policy=None, **kw) -> dict:
        """``batch`` trajectories of ``steps`` steps under ``policy``. See ``stable_envs.rollout``.

        A convenience for when no state corpus exists yet, and available only on the minority of envs
        that ship ``step``. If states already exist, feed them to the renderer from your own loader —
        ``env.set_state(batch).render()`` — which works on the whole registry.
        """
        from .rollout import sample as _sample
        return _sample(self, steps=steps, batch=batch, policy=policy, **kw)

    def variation_values(self) -> dict:
        return self.variation_space.value

    def describe(self) -> dict[str, Any]:
        """Everything a consumer needs to decide whether this env's frames are usable for their job.

        ``fidelity_tier`` is in here, first-class and unconditional, because the answer to "can I train
        on these frames and evaluate on the official ones" is different for the two tiers and there is
        no way to read it off the other fields. A ``"visual"`` env additionally reports its measured
        agreement and the sentence that says what the tier does not license, so a caller who prints
        ``describe()`` and reads nothing else still cannot mistake it for a parity env.
        """
        d = {"name": self.name or type(self).__name__, "state_dim": self.state_dim,
             "state_names": list(self.state_names), "size": self.size,
             "style_dim": self.style_dim, "style_names": self.style_names,
             "reference_id": self.reference_id,
             "fidelity_tier": self.fidelity_tier,
             "render_decoder": self.render_decoder,
             "variations": self.variation_space.names()}
        if self.fidelity_tier == "visual":
            d["visual_fidelity"] = dict(self.visual_fidelity)
            d["visual_note"] = self.visual_note
            d["not_for"] = ("evaluating a policy on official pixels after training on these — the "
                            "frames are a different render of the same scene, not the same frame")
        if self.variation_fidelity:
            d["variation_fidelity"] = dict(self.variation_fidelity)
        rv = reference_versions_of(type(self))
        if rv:
            d[REFERENCE_VERSIONS_KEY] = rv
        if self.render_decoder == "learned" and "not_for" in d:
            # Said separately from ``visual_note`` and in the caller's own words, because this is the
            # one thing about these frames that no amount of looking at them can reveal.
            d["not_for"] += (", and here that different render is a LEARNED approximation: a fixed "
                             "neural checkpoint drew part of it, so detail on textured surfaces is "
                             "synthesised rather than measured")
        else:
            d["parity_interior_min"] = self.parity_interior_min
        return d

    # ------------------------------------------------------------------ batch / device vocabulary

    @property
    def num_envs(self) -> int:
        """How many environments this object holds. ``B`` is the Step-8 spelling and still works."""
        return self.B

    def __len__(self) -> int:
        return self.B

    @property
    def closed(self) -> bool:
        return self._closed

    # ------------------------------------------------------------------ state: packed row <-> State

    def _layout(self) -> tuple[tuple[tuple[str, ...], int, int | None], ...]:
        """This env's ``state_layout``, or the trivial one-key layout when it declares none.

        An instance method, not a classmethod: a family whose state width comes from a compiled
        model (DM Control, Fetch, RoboCasa) only knows ``state_dim`` once it is built.
        """
        if self.state_layout:
            return tuple((tuple(e[0]) if isinstance(e[0], (tuple, list)) else (e[0],), int(e[1]),
                          None if e[2] is None else int(e[2])) for e in self.state_layout)
        return ((("state",), int(self.state_dim), 0),)

    def _layout_dtypes(self) -> dict[tuple[str, ...], torch.dtype]:
        """``{key path: public dtype}`` for the fields whose ``state_layout`` entry names one."""
        out = {}
        for e in self.state_layout or ():
            if len(e) >= 4 and e[3] is not None:
                key = tuple(e[0]) if isinstance(e[0], (tuple, list)) else (e[0],)
                out[key] = e[3] if isinstance(e[3], torch.dtype) else getattr(torch, str(e[3]))
        return out

    def _check_layout(self) -> tuple[tuple[tuple[str, ...], int, int | None], ...]:
        lay = self._layout()
        covered = sum(w for _, w, o in lay if o is not None)
        if covered != int(self.state_dim):
            raise ValueError(
                f"{type(self).__name__}.state_layout covers {covered} of the {self.state_dim} "
                f"column(s) of the packed row. Every packed column belongs to exactly one field, "
                f"because pack/unpack must round-trip the frozen corpus layout unchanged; a field "
                f"that is dynamical state the row does not carry declares offset None instead.")
        return lay

    def pack(self, state: TensorDict) -> torch.Tensor:
        """:data:`State` -> the ``(n, state_dim)`` float32 row every renderer and corpus speaks.

        BYTE FOR BYTE THE STEP-8 LAYOUT. ``state_names`` names its columns and nothing about it moved
        in Step 9; the entity-grouped State is a view of the same numbers with the keys a user would
        pick. Fields whose offset is ``None`` (hidden dynamics that the row never carried, e.g.
        PushT's agent velocity) are dropped here, which is why a packed row is still exactly what the
        frozen digests were computed from.
        """
        if not isinstance(state, TensorDict):
            return check_states(self, state, f"{type(self).__name__}.pack", leading=True)
        lay = self._check_layout()
        n = int(state.batch_size[0]) if state.batch_size else 1
        out = torch.zeros(n, int(self.state_dim), device=self.device, dtype=torch.float32)
        for key, w, off in lay:
            if off is None:
                continue
            try:
                v = state[key]
            except KeyError as e:
                raise KeyError(
                    f"{type(self).__name__}.pack: the state is missing {'.'.join(key)}. This env's "
                    f"state keys are {[ '.'.join(k) for k, _, _ in lay ]}.") from e
            out[:, off:off + w] = torch.as_tensor(v, device=self.device,
                                                  dtype=torch.float32).reshape(n, w)
        return out

    def unpack(self, rows) -> TensorDict:
        """``(n, state_dim)`` packed rows -> :data:`State`.

        Fields the row does not carry come back as ZEROS, which is not a placeholder: it is the
        condition ``set_state`` puts the env in and the condition the reference env is in after
        ``reset``, so ``unpack(pack(s))`` is the same environment and not an approximation of it.
        """
        lay = self._check_layout()
        dtypes = self._layout_dtypes()
        r = check_states(self, rows, f"{type(self).__name__}.unpack").reshape(-1, int(self.state_dim))
        n = int(r.shape[0])
        out: dict = {}
        for key, w, off in lay:
            dt = dtypes.get(key, torch.float32)
            # ``copy=True``: ``.to`` on a same-dtype tensor returns the SAME storage, and a State that
            # aliases the env's packed buffer changes under the caller's feet at the next step.
            v = (r[:, off:off + w].to(dt, copy=True) if off is not None
                 else torch.zeros(n, w, device=self.device, dtype=dt))
            node = out
            for part in key[:-1]:
                node = node.setdefault(part, {})
            node[key[-1]] = v
        return TensorDict(out, batch_size=[n], device=self.device)

    def _hidden_state(self) -> dict[tuple[str, ...], torch.Tensor]:
        """``{key path: (n, width) tensor}`` for the fields whose offset in the packed row is ``None``.

        THE HOOK EXISTS SO THAT A HIDDEN FIELD CANNOT BE FORGOTTEN QUIETLY. A cell that declares a
        field with offset ``None`` is saying "this is real dynamical state that the packed row does
        not carry" — PushT's agent velocity. ``unpack`` fills such a field with zeros, which is right
        for a state loaded from a corpus and WRONG for the env's own live state after a step. A cell
        that overrode the ``state`` property to splice its live value back in would be one forgotten
        override away from a silent wrong answer: ``env.state`` reporting zeros, and
        ``transition(env.state, a) != step(a)`` — a wrong number rather than a crash.

        So the base class asks instead, and REFUSES to guess: declare a hidden field and you must
        answer here. The default returns ``{}``, which is correct for the majority of cells (they
        declare no hidden field at all) and raises for the rest, in ``state``, naming the field.
        """
        return {}

    def _live_state(self) -> dict[tuple[str, ...], torch.Tensor]:
        """``{key path: (n, width) tensor}`` for DRAWN fields whose live value is more precise than
        the packed row.

        The packed float32 row is a lossy projection for a cell that simulates in float64 (classic
        control's claim is near-bit-exactness against a float64 reference): widening the row back
        gives ``float64(float32(x)) != x``, so ``transition(env.state, a)`` would silently diverge
        from ``step(a)`` from the second step on. A cell that keeps a higher-precision live buffer
        answers here for the fields it carries; ``state`` takes these over the widened row. Optional:
        the default is ``{}`` and is correct for every cell whose live state IS the packed row.
        """
        return {}

    @property
    def state(self) -> TensorDict:
        """The state the env currently holds, as a :data:`State`. ``get_state()`` is the packed row.

        The packed half comes from ``unpack(get_state())`` unless :meth:`_live_state` offers a more
        precise live value; the hidden half — the fields the row does not carry — comes from
        :meth:`_hidden_state`, because only the cell knows where it lives.
        """
        td = self.unpack(self.get_state())
        hidden = {tuple(k) if isinstance(k, (tuple, list)) else (k,): v
                  for k, v in (self._hidden_state() or {}).items()}
        n = int(td.batch_size[0])
        dtypes = self._layout_dtypes()
        live = {tuple(k) if isinstance(k, (tuple, list)) else (k,): v
                for k, v in (self._live_state() or {}).items()}
        for key, w, off in self._layout():
            if off is not None and key in live:
                td[key] = torch.as_tensor(live[key], device=self.device,
                                          dtype=dtypes.get(key, torch.float32)).reshape(-1, w)[:n].clone()
        missing = []
        for key, w, off in self._layout():
            if off is not None:
                continue
            if key not in hidden:
                missing.append(".".join(key))
                continue
            td[key] = torch.as_tensor(hidden[key], device=self.device,
                                      dtype=dtypes.get(key, torch.float32)).reshape(-1, w)[:n]
        if missing:
            raise NotImplementedError(
                f"{type(self).__name__}.state: {missing} is declared in state_layout with offset "
                f"None — real dynamical state that the packed row does not carry — and "
                f"{type(self).__name__}._hidden_state() does not return it. Unfilled, it would be "
                f"reported as ZEROS, which is a wrong answer rather than a failure: env.state would "
                f"look fine and transition(env.state, action) would stop matching step(action). "
                f"Return {{{missing[0]!r}: <the live tensor>}} from _hidden_state().")
        return td

    def _step_result(self, state: TensorDict, **extras) -> TensorDict:
        """Build the :data:`StepResult` ``step`` returns: ``{"state": ..., **extras}``.

        Every cell builds it the same way through here rather than writing the ``TensorDict(...)``
        call itself, so the batch size and device cannot drift between cells. ``extras`` are
        ``reward``/``done`` for the cells that HAVE them; a cell with no reward or termination
        semantics passes none, and the keys are absent rather than zero — a fabricated zero reward
        reads as a measured one (contract §5).
        """
        td = {"state": state}
        for k, v in extras.items():
            if v is None:
                continue
            td[k] = torch.as_tensor(v, device=self.device).reshape(int(state.batch_size[0]))
        return StepResult(td, batch_size=[int(state.batch_size[0])], device=self.device)

    def state_names_of(self, key) -> list[str]:
        """Which packed columns a State key occupies. Used by the tests and by ``describe``."""
        key = tuple(key) if isinstance(key, (tuple, list)) else (key,)
        for k, w, off in self._layout():
            if k == key:
                return ([] if off is None else
                        [self.state_names[i] if i < len(self.state_names) else f"column {i}"
                         for i in range(off, off + w)])
        raise KeyError(f"{type(self).__name__}: no state key {'.'.join(key)}")

    # ------------------------------------------------------------------ specs

    @property
    def state_spec(self) -> CompositeSpec:
        """The public state's shape, per entity. Bounds are declared by the env that knows them."""
        bounds = dict(getattr(self, "state_bounds", {}) or {})
        out: dict = {}
        for key, w, _ in self._layout():
            lo, hi = bounds.get(key, (None, None))
            names = tuple(self.state_names_of(key)) if len(key) and w == len(
                self.state_names_of(key)) else ()
            spec = Spec(shape=(w,), low=lo, high=hi, device=self.device, names=names)
            node = out
            for part in key[:-1]:
                node = node.setdefault(part, {})
            node[key[-1]] = spec
        return CompositeSpec(out, device=self.device)

    @property
    def action_spec(self) -> Spec:
        """What ``step``/``transition`` take. Envs with no simulator declare no ``action_dim``."""
        dim = getattr(self, "action_dim", None)
        if dim is None:
            _dynamics.no_dynamics(self, "action_spec")
        return Spec(shape=(int(dim),), low=getattr(self, "action_low", None),
                    high=getattr(self, "action_high", None), device=self.device)

    @property
    def observation_spec(self) -> Spec:
        """The frames ``render`` produces: ``(H, W, 3)`` uint8. Square unless the env says otherwise."""
        h, w = getattr(self, "frame_shape", (self.size, self.size))
        return Spec(shape=(int(h), int(w), 3), dtype=torch.uint8, low=0, high=255,
                    device=self.device)

    @property
    def variation_spec(self) -> Spec:
        """The flat style layout — ``style_dim`` columns named by ``style_names``.

        FLAT, and a single ``Spec`` rather than a composite, because that is what a style tensor IS:
        one ``(B, style_dim)`` row whose columns are addressed by path. ``variation_space`` remains
        the gymnasium-style object the certification harness samples and validates against; this is
        the same layout in the package's own vocabulary.
        """
        lo, hi, names = [], [], []
        for path, off, w in self.variation_space.flat_layout():
            # ONE NAME PER COLUMN, which ``style_names`` is not: a leaf of width 3 (an RGB colour)
            # occupies three columns under one path, and a spec whose names do not line up with its
            # columns is a labelling that would be wrong exactly where it is consulted.
            names.extend([path] if w == 1 else [f"{path}[{i}]" for i in range(w)])
            node = self.variation_space
            for part in path.split("."):
                node = node[part]
            if hasattr(node, "low"):
                lo.extend(np.broadcast_to(node.low, node.shape).astype(np.float64).reshape(-1))
                hi.extend(np.broadcast_to(node.high, node.shape).astype(np.float64).reshape(-1))
            elif hasattr(node, "n"):
                lo.extend([0.0] * w)
                hi.extend([float(node.n) - 1.0] * w)
            else:
                lo.extend([-np.inf] * w)
                hi.extend([np.inf] * w)
        return Spec(shape=(int(self.style_dim),), low=np.asarray(lo, np.float64),
                    high=np.asarray(hi, np.float64), device=self.device, names=tuple(names))

    # ------------------------------------------------------------------ lifecycle

    def reset(self, seed: int | None = None) -> TensorDict:
        """Draw ``num_envs`` states from the env's own reset distribution and hold them.

        ``seed`` re-seeds the env's generator, so ``reset(seed=0)`` twice gives the same states — the
        determinism the whole corpus story rests on. With no seed the generator continues, which is
        what a sequence of resets inside one experiment should do.
        """
        self._check_open("reset")
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
            self._seed = int(seed)
        self.set_state(self.sample_states(self.B, self.rng))
        return self.state

    def step(self, actions) -> TensorDict:
        """Advance every environment one control step. STATEFUL: it writes ``env.state``.

        Returns a :data:`StepResult` holding at least ``"state"``, built with ``_step_result``. A
        cell with no reward or termination semantics omits those keys rather than fabricating zeros
        (contract §5).
        """
        _dynamics.no_dynamics(self, "step")

    def transition(self, state: TensorDict, action) -> TensorDict:
        """``(state, action) -> next_state``, PURE: the env's own buffers are not touched."""
        _dynamics.no_dynamics(self, "transition")

    def unroll(self, state: TensorDict, actions) -> TensorDict:
        """``actions`` are ``(T, num_envs, ...)`` and KNOWN — open loop, never a policy rollout."""
        return _dynamics.unroll_from_transition(self, state, actions)

    @property
    def has_dynamics(self) -> bool:
        """True when this env ships a batched simulator, i.e. overrides ``step`` or ``transition``."""
        return _dynamics.has_dynamics(self)

    def _check_open(self, what: str) -> None:
        if self._closed:
            raise RuntimeError(
                f"{type(self).__name__}.{what}: this environment is closed. close() releases the "
                f"renderer's resources (contexts, atlases, device buffers), so what came back would "
                f"be drawn through something that no longer exists — or, worse, would look like a "
                f"frame. Build a new env: {type(self).__name__}(num_envs={self.B}, "
                f"device='{self.device.type}').")

    def close(self) -> None:
        """Release this env's resources. IDEMPOTENT, and correctness never depends on ``__del__``."""
        if self._closed:
            return
        self._closed = True
        closer = getattr(self, "_close", None)
        if callable(closer):
            closer()

    def __enter__(self) -> "Env":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False


#: ``dynamics.has_dynamics`` asks whether a class OVERRIDES ``step``/``transition``, and it needs the
#: class they are inherited from to compare against. Set here rather than inside ``dynamics`` because
#: ``env`` imports ``dynamics`` and not the other way round: the arrow stays one-way.
Env._se_dynamics_base = Env
