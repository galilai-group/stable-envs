"""The Overcooked cases of the release gate's exclusions, moved from ``tests/regression/test_declarations.py``.

``tests/_runner/gate.py`` still excludes Overcooked by name; the module those exclusions point at is
now ``contrib/overcooked.py``.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_gate_envs_removes_exactly_the_excluded_and_nothing_else():
    """``gate_envs`` is subtraction, not filtering-by-availability."""
    from tests._runner import gate

    names = ["PushT", "OvercookedCrampedRoom", "DMCWalkerWalk", "OvercookedCoordRing"]
    assert gate.gate_envs(names) == ["DMCWalkerWalk", "PushT"]
    assert gate.is_excluded("PushT") is None
    assert gate.is_excluded("OvercookedCrampedRoom")
    # An excluded env is still a first-class env: the carve-out is about which INTERPRETER runs the
    # gate, never about removing the family. If this ever fails, someone deleted an env believing the
    # exclusion licensed it.
    assert (ROOT / "contrib" / "overcooked.py").exists()


def test_the_excluded_family_documents_itself_where_a_user_would_look():
    """The exclusion must be in the module's own docstring, not only in the gate's table.

    Someone hitting this reads the env module or its ImportError; they do not read the test harness.
    """
    src = (ROOT / "contrib" / "overcooked.py").read_text()
    # Case-insensitive: this is prose, and the first version of this test asserted "own interpreter"
    # against a docstring that shouts "OWN INTERPRETER". A guard that fails on capitalisation trains
    # people to edit the guard, which is the opposite of what it is for.
    head = src[: src.index('"""', 3) + 3].lower()
    for token in ("scipy", "own interpreter", "gate.py"):
        assert token in head, (
            f"contrib/overcooked.py's module docstring does not mention {token!r}. The "
            f"family is excluded from the release gate, and a user who hits that must be able to "
            f"find out why from the module itself."
        )
