"""The render corrector's REFUSALS, on CPU, with no env and no checkpoint of anyone's needed.

Everything here is a way of asking one question: can a corrector end up applied to something it was
not fitted on, or described by numbers that were not measured on it? Both failures are silent — a
mismatched corrector produces a confident, plausible, wrong picture, and it does not lower the pixel
mean enough to notice (§4 of ``scratch/altrender/rendernet/README.md``: on the wrong camera it is 2.6x
worse than a plain resize, and on a different kitchen the MEAN ties while SSIM and within-8 fall).

So these are the tests that must not need a GPU, a cache directory or a trained network to run, and
they build their own two-parameter checkpoints to make sure of it. ``tests/gpu/test_rendernet.py``
asserts the same properties on the real thing.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from stable_envs.render import rendernet as RN

NET = "espcn_112t128_w48_d4_u2"
CELL_A = {
    "env_class": "AlphaCorrected",
    "family": "robocasa",
    "scene": "L52S03",
    "camera": "robot0_agentview_left",
    "scene_sha": "0123456789abcdef",
}
CELL_B = dict(CELL_A, env_class="BetaCorrected", camera="robot0_agentview_right")
RENDER = {"render_size": 112, "out_size": 256, "tex_lod": 2, "backface_cull": False}
VF = {
    "mean_abs": 2.45,
    "within8_frac": 0.935,
    "ssim": 0.97,
    "mean_abs_max": 3.2,
    "within8_min": 0.88,
    "ssim_min": 0.92,
    "measured_at": "a test",
    "decoder": NET,
}


def _seal(tmp_path, cell=CELL_A, net=NET, **over):
    m = RN.build(net)
    kw = dict(
        cell=cell,
        render=RENDER,
        net=net,
        state_dict=m.state_dict(),
        visual_fidelity=VF,
        variation_fidelity=VF,
        baseline={"mean_abs": 6.41},
        provenance={"test": True},
    )
    kw.update(over)
    return RN.seal(str(tmp_path / f"{cell['env_class']}.pt"), **kw)


def test_a_sealed_corrector_loads_for_the_cell_it_was_sealed_for(tmp_path):
    p = _seal(tmp_path)
    c = RN.load(CELL_A, device="cpu", path=p)
    assert c.cell == CELL_A
    assert (c.render_size, c.out_size, c.tex_lod, c.backface_cull) == (112, 256, 2, False)
    assert c.visual_fidelity["mean_abs"] == 2.45


@pytest.mark.parametrize(
    "field,value",
    [
        ("camera", "robot0_eye_in_hand"),
        ("scene", "L45S33"),
        ("scene_sha", "deadbeefdeadbeef"),
        ("env_class", "SomethingElseCorrected"),
        ("family", "humanoidbench"),
    ],
)
def test_a_corrector_refuses_every_cell_it_was_not_fitted_on(tmp_path, field, value):
    """THE property. One field wrong is enough, and it must be fatal rather than approximate.

    Each of these five has a measured cost. A different camera on the same kitchen is 5.94/255 against
    6.14 for a plain resize and 4.75 for the render being replaced — i.e. no better than doing
    nothing. A wrist camera is 11.35 against 4.39, 2.6x worse. A different kitchen ties on the mean
    and loses on SSIM (0.763 against 0.792) because the network paints marble-like speckle onto plain
    cabinet fronts. A rebuilt scene (``scene_sha``) is a kitchen whose geometry these weights have
    never seen. And the class name is what a user typed.
    """
    p = _seal(tmp_path)
    with pytest.raises(RN.CellMismatch) as e:
        RN.load(dict(CELL_A, **{field: value}), device="cpu", path=p)
    assert field in str(e.value), "the refusal must name the field that disagreed"
    assert "per (scene, camera)" in str(e.value)


def test_two_cells_cannot_share_one_checkpoint_by_renaming_the_file(tmp_path):
    """The lock is the CELL inside the file, not the filename the loader found it under.

    Copying ``AlphaCorrected.pt`` to ``BetaCorrected.pt`` is the one-line way a well-meaning person
    gives a cell "a corrector to start from", and it is exactly the configuration §4 measured as
    worse than not correcting at all.
    """
    p = _seal(tmp_path)
    import shutil

    q = str(tmp_path / "BetaCorrected.pt")
    shutil.copy(p, q)
    shutil.copy(p[:-3] + ".json", q[:-3] + ".json")
    with pytest.raises(RN.CellMismatch):
        RN.load(CELL_B, device="cpu", path=q)


def test_an_edited_sidecar_declaration_is_refused(tmp_path):
    """A floor lowered in the json a reader sees, without touching the weights, does not load.

    The sidecar exists so a declaration can be read without deserialising a network; making it
    authoritative would have made it the easiest thing in the repository to weaken.
    """
    p = _seal(tmp_path)
    side = p[:-3] + ".json"
    with open(side) as f:
        hdr = json.load(f)
    hdr["visual_fidelity"]["mean_abs_max"] = 99.0
    with open(side, "w") as f:
        json.dump(hdr, f)
    with pytest.raises(ValueError, match="disagrees with the header"):
        RN.load(CELL_A, device="cpu", path=p)


def test_a_declaration_measured_on_other_weights_is_refused(tmp_path):
    """Swap the weights, keep the header: the numbers now describe a checkpoint that is not here."""
    p = _seal(tmp_path)
    ck = torch.load(p, map_location="cpu", weights_only=True)
    ck["state_dict"]["head.bias"] = ck["state_dict"]["head.bias"] + 1.0
    torch.save(ck, p)
    with pytest.raises(ValueError, match="different weights"):
        RN.load(CELL_A, device="cpu", path=p)


def test_a_checkpoint_from_another_format_is_refused_rather_than_guessed(tmp_path):
    p = _seal(tmp_path)
    ck = torch.load(p, map_location="cpu", weights_only=True)
    ck["format"] = "stable-envs/rendernet/0"
    torch.save(ck, p)
    with pytest.raises(ValueError, match="format"):
        RN.load(CELL_A, device="cpu", path=p)


def test_a_missing_checkpoint_says_how_to_build_one(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        RN.load(CELL_A, device="cpu", path=str(tmp_path / "nothing.pt"))
    msg = str(e.value)
    assert "train_corrector.py" in msg, "an absent corrector must name the command that makes one"
    assert "code, not weights" in msg


def test_listing_an_empty_cache_is_empty_and_not_an_error(tmp_path, monkeypatch):
    """No checkpoints anywhere is the DEFAULT state of a fresh clone, and it must be quiet.

    The whole opt-in mechanism rests on this: with nothing in the cache, no corrected class is built,
    every plain visual-tier class is untouched, and nothing warns about a feature nobody asked for.
    """
    monkeypatch.setattr(RN, "CACHE_ROOT", str(tmp_path / "does-not-exist"))
    assert RN.available() == {}


def test_listing_skips_files_that_are_not_ours(tmp_path, monkeypatch):
    monkeypatch.setattr(RN, "CACHE_ROOT", str(tmp_path))
    (tmp_path / "notes.json").write_text("{}")
    (tmp_path / "orphan.json").write_text(json.dumps({"format": RN.FORMAT, "cell": {}}))
    _seal(tmp_path)
    assert sorted(RN.available()) == ["AlphaCorrected"]


def test_the_network_starts_exactly_at_the_bilinear_baseline():
    """A freshly built corrector IS the bilinear upsample, to the byte.

    This is the honesty property the zero-initialised head buys, and it is worth a test because it is
    what makes every trained number a departure from a point that was already measured: a run that
    fails degrades to a known baseline rather than to noise, and the residual a corrector adds is a
    tensor somebody can look at.
    """
    import torch.nn.functional as F

    m = RN.build(NET).eval()
    u8 = torch.randint(0, 256, (2, 112, 112, 3), dtype=torch.uint8)
    got = m(u8)
    # Through the module's OWN preprocessing, because ``F.interpolate`` on a channels_last tensor and
    # on a contiguous one are different kernels and disagree by an ulp, which lands on the far side of
    # a rounding for about one pixel in 100k. That is a fact about the resize, not about the head, and
    # this test is about the head.
    x = u8.permute(0, 3, 1, 2).float().mul_(1.0 / 255.0).contiguous(memory_format=torch.channels_last)
    anchor = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
    assert float(m.head.weight.abs().max()) == 0.0 and float(m.head.bias.abs().max()) == 0.0, (
        "the residual head is not zero-initialised, so training would start somewhere unmeasured"
    )
    ref = anchor.mul_(255.0).round_().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
    assert torch.equal(got, ref)


def test_an_unknown_architecture_is_refused_by_name():
    with pytest.raises(KeyError, match="unknown corrector architecture"):
        RN.build("espcn_made_up")
