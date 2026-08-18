"""The standalone claims in ``stable_envs/envs/swm_nav.py``'s docstring, measured on the reference.

Kept apart from the parity run so each number has an obvious provenance and can be re-checked in
about twenty seconds:

  1. the Gaussian "dot" is a 40-pixel object, not a 7-pixel one, and the truncation is what makes it
     one — measured on the OFFICIAL frame, with the wall and border pixels excluded so the argmin
     finds the agent rather than the black border (which it did on the first attempt);
  2. ``wall.border_color`` is declared by the official env and read by nothing;
  3. the border stroke is 4 px at rows/cols 10..13 and 210..213;
  4. the two alpha blends in ``TwoRoomEnv._render_frame`` are separated by a uint8 quantisation, so
     chaining them in float32 is a real error rather than a stylistic choice.
"""
from __future__ import annotations

import json

import numpy as np
import torch

import stable_envs as se
from stable_envs.testing import conformance as C
from stable_envs.envs import swm_nav as M


def facts() -> dict:
    import gymnasium
    import stable_worldmodel  # noqa: F401

    out = {}
    env = gymnasium.make("swm/TwoRoom-v1")
    u = env.unwrapped
    env.reset(seed=0, options={"variation": []})
    u.agent_position = torch.tensor([60.0, 112.0])
    f0 = np.ascontiguousarray(u.render())

    # The agent is (255,0,0) on white, so every pixel it touches keeps R == 255 while G falls from
    # 255 to 0. The wall and border are (0,0,0), i.e. R == 0, which is how they are excluded — the
    # first version of this probe took argmin(G) over the whole frame and reported the border corner.
    r, g = f0[..., 0].astype(np.int16), f0[..., 1].astype(np.int16)
    agentish = r == 255
    gg = np.where(agentish, g, 255)
    ay, ax = np.unravel_index(int(np.argmin(gg)), gg.shape)
    out["agent_pos"] = [60.0, 112.0]
    out["agent_peak_rc"] = [int(ay), int(ax)]
    out["agent_peak_rgb"] = [int(v) for v in f0[ay, ax]]
    touched = agentish & (g < 255)
    ys, xs = np.nonzero(touched)
    out["touched_px"] = int(touched.sum())
    out["halo_254_px"] = int((agentish & (g == 254)).sum())
    out["distinct_green_levels"] = int(np.unique(gg).size)
    out["max_touched_radius"] = float(np.sqrt((ys - ay) ** 2 + (xs - ax) ** 2).max())
    out["nominal_disc_px_r7"] = int((((np.arange(224)[:, None] - ay) ** 2
                                      + (np.arange(224)[None, :] - ax) ** 2) <= 49).sum())

    u.variation_space["wall"]["border_color"].set_value(np.array([255, 255, 255], np.uint8))
    fw = np.ascontiguousarray(u.render())
    u.variation_space["wall"]["border_color"].set_value(np.array([0, 0, 0], np.uint8))
    fb = np.ascontiguousarray(u.render())
    out["border_color_max_abs_diff"] = int(np.abs(fw.astype(np.int16) - fb.astype(np.int16)).max())

    col = f0[:, 5]
    out["border_rows_at_x5"] = [int(v) for v in
                                np.nonzero((col != np.array([255, 255, 255])).any(-1))[0]]
    return out


def blend_order_cost() -> dict:
    """What does chaining the two blends in float32 cost, against quantising between them?

    Both variants are compared to the OFFICIAL frames on the same states, so the answer is not "the
    two of ours differ" (which proves nothing about which is right) but "one of them is the
    reference's and the other is not".
    """
    cls = se.REGISTRY["TwoRoomTarget"]
    env = se.make("TwoRoomTarget", batch=16, device="cuda")
    ref = C.reference_env(cls, env.size)
    st = env.sample_states(64, rng=np.random.default_rng(3))
    official = cls.reference_render(ref, st)
    correct = env.render(st).cpu().numpy()

    original = M._alpha_blend

    def float32_chained(img, alpha, rgb):
        """The tempting version: stay in float32 and quantise once at the end."""
        a = alpha.clamp(0.0, 1.0).unsqueeze(-1)
        out = img.to(torch.float32) if img.dtype == torch.uint8 else img
        return out * (1.0 - a) + M._trunc_color(rgb).view(-1, 1, 1, 3) * a

    M._alpha_blend = float32_chained
    naive = env.render(st)
    M._alpha_blend = original
    naive = naive.cpu().numpy()

    d_ok = np.abs(correct.astype(np.int16) - official.astype(np.int16))
    d_no = np.abs(naive.astype(np.int16) - official.astype(np.int16))
    return {"quantise_between_bad_px": int((d_ok.max(-1) > 0).sum()),
            "float32_chained_bad_px": int((d_no.max(-1) > 0).sum()),
            "float32_chained_max_abs": int(d_no.max()),
            "n_pixels": int(d_ok[..., 0].size)}


if __name__ == "__main__":
    out = {"reference_facts": facts()}
    try:
        out["blend_order"] = blend_order_cost()
    except Exception as e:                                                  # noqa: BLE001
        import traceback
        out["blend_order"] = traceback.format_exc()
    print(json.dumps(out, indent=1, default=str))
    with open("/mnt/vast/home/randall.balestriero/stable_envs_logs/agentB/swm_nav_facts.json",
              "w") as f:
        json.dump(out, f, indent=1, default=str)
