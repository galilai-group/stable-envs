# Blender G34 nominal geometry oracle

Source: Hokin Deng, object-permanence, commit
`824baddcdad087810d35638bf8f273b884f74975`, G34 authored nominal scene.

The archive contains 120 evaluated ball poses and 21 independent Blender
triangle-raycast depth/visible-ID maps at 320x180. It is not a MuJoCo checkpoint,
an EEVEE RGB recording, or a randomized dataset sample. Camera, geometry, version,
units, identity mapping, array schema and hashes are in `manifest.json`.

Generated only by `scripts/testing/freeze_occlusion_ball_oracle.py`, which
imports the unmodified pinned source in Blender 4.4 and never imports the
candidate environment. Existing output cannot be overwritten. Reproduction and
acceptance details are in `docs/adding/occlusion-ball.md`.

These derived data are attributed to the source under CC BY-NC 4.0. Full notice
and terms are in `SOURCE_LICENSE.txt`; this fixture is not relicensed under the
surrounding repository's MIT license.
