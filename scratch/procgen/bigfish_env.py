"""bigfish, rendered bit-exactly and NOT shipped — the spawner-cast boundary, in code.

This is a complete, measured procgen renderer: **240 frames over 20 levels x 12 steps, 100.0000% of
pixels exact, max |diff| 0** against the shipped Qt renderer (`probe18_bigfish_parity.py`), with
every fish theme, both reflections, radii from 0.2505 to 1.9936, five background indices and
`bg_pct_x` from 0.010 to 0.967 exercised. The renderer is right.

It does not ship because the *state contract* cannot be met, and the two requirements that collide
are both measurable (`docs/research/pixel_native.md` 2.9):

  * a fixed-width slot state must not silently drop an entity, so the cap has to be at least the
    reachable maximum — **24 fish at once**, over 12,800 states harvested from 32 parallel envs
    (`probe17_bigfish.py`);
  * every declared state dimension must change the frame, and slot `i`'s coordinates only do so when
    slot `i` is occupied. Measured: the shipped pool holds **mean 4.3 / max 10** fish, so slots 7+
    are dead at B=8 and 9+ at B=32 (`probe20_bigfish_deaddims.py`); even a steady-state pool tops out
    near 15, because a 24-fish state occurs **once in 12,800**.

So the cap must be >= 24 and <= about 15, and no value satisfies both. chaser ships and bigfish does
not for exactly this reason: chaser's cast is a constant of the game (3 orbs, 3 eggs, 3 enemies),
bigfish's is a Bernoulli(1/10) spawner. That distinction, not per-game draw complexity, is what
decides the remaining procgen games.

Kept runnable rather than deleted so the parity number above stays reproducible and so the boundary
can be re-tested if the contract ever grows a variable-length state.

    PYTHONPATH=. python scratch/procgen/probe18_bigfish_parity.py
"""
from __future__ import annotations

import numpy as np
import torch

from stable_envs.envs.procgen import (  # noqa: F401
    P_SPACE, SPECS, WATER_BG, _EntSlot, _ProcgenEnv, _Spec, _CLASS_OF, _PROCGEN_ID, _make_entity)

#: bigfish: no grid reaches a pixel, and the cast is not fixed. `BIGFISH_SLOTS` is the cap.
SPECS["bigfish"] = _Spec(
    "bigfish", 20, ("space",),
    ["misc_assets/fishTile_072.png",           # 0  PLAYER    108x76
     "misc_assets/fishTile_074.png",           # 1  FISH t0    78x46
     "misc_assets/fishTile_078.png",           # 2  FISH t1   112x88
     "misc_assets/fishTile_080.png"],          # 3  FISH t2   108x58
    WATER_BG,
    None,                                      # filled in below: BIGFISH_SLOTS fish, then agent
    [-1])
#: How many fish a fixed-width state carries. **Measured, not chosen**: 12,800 states harvested from
#: 32 parallel envs over 2,000 steps hold at most **24** fish at once, with a p99 of 13 and a tail of
#: {17: 2, 20: 1, 21: 1, 24: 1} (`probe17_bigfish.py`). 32 is that maximum with a third again of
#: headroom. A state holding more is outside the contract and `state_from_blob` says so rather than
#: silently dropping the overflow, because a dropped fish is a frame that is wrong in a way no
#: parity number taken on the pool would show. The dead-dimension cost of that headroom is what
#: keeps this env out of the registry: `probe20_bigfish_deaddims.py` reports slots 7+ dead at B=8.
BIGFISH_SLOTS = 32
SPECS["bigfish"].ent_slots = (
[_EntSlot(f"fish{i}", 1, 0.5, 0.5, z=0, reflect=True) for i in range(BIGFISH_SLOTS)]
+ [_EntSlot("agent", 0, 0.5, 0.5, z=1, reflect=True)])
#: `asset_for_type(FISH)` names three sprites, so `image_theme` indexes them directly; the agent's
#: is a separate type with one. Atlas slots for themes 0, 1, 2.
_BIGFISH_FISH_SPRITE = (1, 2, 3)

_CLASS_OF["bigfish"] = {P_SPACE: 0}
_PROCGEN_ID["bigfish"] = [P_SPACE]

class ProcgenBigfish(_ProcgenEnv):
    """Eat smaller fish, avoid bigger ones. The first procgen game here with a variable cast.

    Everything maze/heist/miner/chaser could take as a constant of the slot is state here, and each
    of the three was measured (``scratch/procgen/probe17_bigfish.py``):

      * **the cast is not fixed.** A fish spawns with probability 1/10 per step and is erased when it
        leaves the world, so a fixed-width state has to cap the count. 12,800 harvested states hold
        at most **24** at once (p99 13); ``BIGFISH_SLOTS = 32``. This is the reason the class is not
        ``@register``ed — see this module's own docstring.
      * **radius and sprite are per entity.** Fish are sampled ``rx`` in [0.25, 2] and given one of
        three sprites by ``choose_random_theme``, then ``match_aspect_ratio`` sets
        ``ry = rx / (w/h)`` — measured at 1.69565 / 1.27273 / 1.86207, which is exactly 78/46,
        112/88 and 108/58. The three sprites are DIFFERENT SIZES, so the blit's source extent is per
        environment too. The agent's radius grows by ``r_inc`` per fish eaten.
      * **the z=0 draw order IS observable**, which it was not for heist or chaser: fish overlap, and
        reversing the z=0 entity list changed **4 of 166** real frames (22 of which had two z=0
        entities overlapping in world space at all). So slot ``i`` is the ``i``-th fish in the
        entity list — spawn order — and not a sort. This is the first game in the family where that
        distinction is load-bearing.

    The grid is never written (``game_reset`` leaves it all SPACE and ``draw_grid_obj`` returns on
    SPACE), so ``has_grid`` is False and the 400 cells are not state. ``bg_pct_x`` IS state: none of
    the seven ``water_backgrounds`` is square.
    """

    game = "bigfish"
    name = "procgen-bigfish"
    reference_id = "procgen-bigfish-v0"
    has_grid = False
    state_dim = 2 + 4 + BIGFISH_SLOTS * 7
    state_names = ("bg_index", "bg_pct_x", "agent_x", "agent_y", "agent_r",
                   "agent_reflected") + tuple(
        f"fish{i}_{c}" for i in range(BIGFISH_SLOTS)
        for c in ("x", "y", "rx", "ry", "theme", "reflected", "present"))
    _BASE = 6

    def _unpack(self, s):
        f = super()._unpack(s)
        f["bg_pct_x"] = s[:, 1]
        f["agent"] = (s[:, 2], s[:, 3], s[:, 4], s[:, 5])
        b = self._BASE
        f["fish"] = [tuple(s[:, b + 7 * i + k] for k in range(7)) for i in range(BIGFISH_SLOTS)]
        return f

    def _slot_fields(self, f, sl, B):
        if sl.key == "agent":
            x, y, _, refl = f["agent"]
            return None, x, y, None, refl
        x, y, _, _, _, refl, on = f["fish"][int(sl.key[4:])]
        return on > 0.5, x, y, None, refl

    def _slot_radii(self, f, sl, B):
        # clamped away from 0 only so the blit's `src_extent / extent` stays finite on states no
        # rollout reaches. It cannot change a frame: at 1e-4 the rect is 6.4e-4 device px wide and
        # `QRectF::toRect()` gives it zero width, which is what a zero radius gave anyway.
        if sl.key == "agent":
            r = f["agent"][2].clamp_min(1e-4)
            return r, r
        _, _, rx, ry, _, _, _ = f["fish"][int(sl.key[4:])]
        return rx.clamp_min(1e-4), ry.clamp_min(1e-4)

    _sprite_lut: dict = {}

    def _slot_sprite(self, f, sl, B):
        if sl.key == "agent":
            return None
        key = str(self.device)
        lut = self._sprite_lut.get(key)
        if lut is None:
            lut = torch.as_tensor(_BIGFISH_FISH_SPRITE, device=self.device)
            self._sprite_lut[key] = lut
        return lut[f["fish"][int(sl.key[4:])][4].round().clamp_(0, 2).long()]

    @classmethod
    def state_from_blob(cls, b):
        ents = b["entities"]
        a = [e for e in ents if e["type"] == 0][0]
        fish = [e for e in ents if e["type"] == 2]                 # FISH, in entity-list order
        if len(fish) > BIGFISH_SLOTS:
            raise AssertionError(
                f"ProcgenBigfish: {len(fish)} fish in one state, above the declared "
                f"BIGFISH_SLOTS = {BIGFISH_SLOTS}. The cap was measured at 24 over 12,800 states; "
                f"raise BIGFISH_SLOTS and re-record state_dim rather than dropping the overflow.")
        row = [b["background_index"], b["bg_pct_x"], a["x"], a["y"], a["rx"], a["is_reflected"]]
        for i in range(BIGFISH_SLOTS):
            e = fish[i] if i < len(fish) else None
            # An unoccupied slot carries a LEGAL radius rather than 0. `present = 0` masks it out of
            # the frame either way, but 0 makes `src_extent / extent` a division by zero inside the
            # blit map, and a masked NaN is a trap waiting for the first person to change the mask.
            row += ([e["x"], e["y"], e["rx"], e["ry"], e["image_theme"], e["is_reflected"], 1.0]
                    if e else [0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0])
        return np.asarray(row, np.float32)

    @classmethod
    def blob_from_state(cls, template, st):
        b = dict(template)
        b["grid"] = np.full((20, 20), P_SPACE, np.int32)
        b["grid_w"] = b["grid_h"] = 20
        b["grid_size"] = 400
        b["background_index"] = int(round(float(st[0])))
        b["bg_pct_x"] = float(st[1])
        S = {s.key: s for s in SPECS["bigfish"].ent_slots}
        ents = []
        for i in range(BIGFISH_SLOTS):
            x, y, rx, ry, theme, refl, on = st[cls._BASE + 7 * i:cls._BASE + 7 * i + 7]
            if on > 0.5:
                e = _make_entity(S[f"fish{i}"], x, y, int(round(float(theme))), 2, 2,
                                 refl=int(round(float(refl))))
                e["rx"], e["ry"] = float(rx), float(ry)     # per-entity, not the slot's constant
                ents.append(e)
        ag = _make_entity(S["agent"], st[2], st[3], 0, 0, 0,
                          refl=int(round(float(st[5]))), agent=True)
        ag["rx"] = ag["ry"] = float(st[4])
        ents.append(ag)
        b["entities"] = ents
        return b


